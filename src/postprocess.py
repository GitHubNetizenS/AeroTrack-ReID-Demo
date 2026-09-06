"""
postprocess.py文件
　　该文件负责对跟踪与OCR结果进行最终清洗：车牌纠错、同车合并、伪车辆过滤，以及使用车牌号作为目录名。
"""

import re       as re
from typing     import Any, Dict, List, Optional


CHINESE_PLATE_PATTERN = re.compile(r"^[\u4e00-\u9fa5][A-Z][A-Z0-9]{5,6}$")


def plate_suffix(plate_text: str) -> str:
    if not plate_text:
        return ""
    plate_text = plate_text.upper().replace(" ", "").replace("-", "")
    if plate_text and "\u4e00"<=plate_text[0]<="\u9fff":
        return plate_text[1:]
    return plate_text


def levenshtein_distance(left: str, right: str) -> int:
    if left==right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right)+1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char==right_char else 1
            current.append(min(
                current[j-1]+1,
                previous[j]+1,
                previous[j-1]+cost,
            ))
        previous = current
    return previous[-1]


def normalize_expected_plates(plates: List[str]) -> List[str]:
    normalized = []
    for plate in plates:
        plate = plate.upper().replace(" ", "").replace("-", "")
        if plate and plate not in normalized:
            normalized.append(plate)
    return normalized


def normalize_plate_aliases(plate_aliases: Dict[str, str]) -> Dict[str, str]:
    normalized = {}
    for key, value in plate_aliases.items():
        norm_key = str(key).upper().replace(" ", "").replace("-", "")
        norm_value = str(value).upper().replace(" ", "").replace("-", "")
        normalized[norm_key] = norm_value
    return normalized


def canonicalize_plate(plate_text: str, expected_plates: List[str], max_edit_distance: int = 2,
                       default_province: str = "", default_city_letter: str = "", plate_aliases: Optional[Dict[str, str]] = None,
                       allow_unmatched_legal: bool = False) -> Optional[str]:
    plate_text = plate_text.upper().replace(" ", "").replace("-", "")
    if not plate_text or plate_text=="UNKNOWN":
        return None

    plate_aliases = normalize_plate_aliases(plate_aliases or {})
    if plate_text in plate_aliases:
        return plate_aliases[plate_text]

    expected_plates = normalize_expected_plates(expected_plates)
    if not expected_plates:
        if CHINESE_PLATE_PATTERN.match(plate_text):
            return plate_text
        if default_province and default_city_letter and plate_text.startswith(default_city_letter):
            candidate = f"{default_province}{plate_text}"
            if CHINESE_PLATE_PATTERN.match(candidate):
                return candidate
        return None

    if plate_text in expected_plates:
        return plate_text

    candidate_suffix = plate_suffix(plate_text)
    if not candidate_suffix:
        return None

    for expected in expected_plates:
        if candidate_suffix==plate_suffix(expected):
            return expected

    best_plate = None
    best_distance = 999
    for expected in expected_plates:
        expected_suffix = plate_suffix(expected)
        distance = levenshtein_distance(candidate_suffix, expected_suffix)
        if distance<best_distance:
            best_distance = distance
            best_plate = expected

    if best_plate is not None and best_distance<=max_edit_distance:
        return best_plate
    if allow_unmatched_legal:
        if CHINESE_PLATE_PATTERN.match(plate_text):
            return plate_text
        if default_province and default_city_letter and plate_text.startswith(default_city_letter):
            candidate = f"{default_province}{plate_text}"
            if CHINESE_PLATE_PATTERN.match(candidate):
                return candidate
    return None


def is_near_duplicate_plate(left: str, right: str, max_distance: int) -> bool:
    if not left or not right or left.startswith("UNKNOWN") or right.startswith("UNKNOWN"):
        return False
    return levenshtein_distance(plate_suffix(left), plate_suffix(right))<=max_distance


def consolidate_vehicle_records(records: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    post_config = config.get("postprocess", {})
    use_expected_plates = post_config.get("use_expected_plates", False)
    soft_expected_plate_matching = post_config.get("soft_expected_plate_matching", False)
    expected_plates = post_config.get("expected_plates", []) if (use_expected_plates or soft_expected_plate_matching) else []
    allow_unmatched_legal = bool(soft_expected_plate_matching and not use_expected_plates)
    max_edit_distance = post_config.get("plate_max_edit_distance", 2)
    keep_unknown = post_config.get("keep_unknown", False)
    min_crop_count = post_config.get("min_crop_count", 1)
    default_province = post_config.get("default_plate_province", "")
    default_city_letter = post_config.get("default_plate_city_letter", "")
    plate_aliases = post_config.get("plate_aliases", {})
    manual_track_plate_overrides = {}
    for track_id, override in post_config.get("manual_track_plate_overrides", {}).items():
        if isinstance(override, dict):
            manual_track_plate_overrides[int(track_id)] = {
                "plate": str(override.get("plate", "")),
                "start_frame": override.get("start_frame"),
                "end_frame": override.get("end_frame"),
            }
        else:
            manual_track_plate_overrides[int(track_id)] = {
                "plate": str(override),
                "start_frame": None,
                "end_frame": None,
            }
    merge_near_duplicate_plate_text = post_config.get("merge_near_duplicate_plate_text", True)
    near_duplicate_max_distance = post_config.get("near_duplicate_max_distance", 1)
    near_duplicate_small_group_max_crops = post_config.get("near_duplicate_small_group_max_crops", 2)
    assign_unobserved_crops = post_config.get("assign_unobserved_crops_to_dominant_plate", True)
    observation_neighbor_crops = post_config.get("observation_neighbor_crops", 0)
    drop_minor_conflicting_observations = post_config.get("drop_minor_conflicting_plate_observations", False)
    dominant_plate_min_votes = post_config.get("dominant_plate_min_votes", 2)
    dominant_plate_score_ratio = post_config.get("dominant_plate_score_ratio", 1.25)
    groups: Dict[str, Dict[str, Any]] = {}
    unknown_index = 1

    def get_record_crop_paths(record: Dict[str, Any]) -> List[str]:
        rejected = set(record.get("rejected_crop_paths", []))
        return [
            crop_path for crop_path in record.get("crop_paths", [])
            if crop_path not in rejected
        ]

    def get_group(canonical_plate: str, record: Dict[str, Any]) -> Dict[str, Any]:
        group = groups.get(canonical_plate)
        if group is None:
            group = {
                "vehicle_id": canonical_plate,
                "track_ids": [],
                "plate_text": canonical_plate if not canonical_plate.startswith("UNKNOWN") else "UNKNOWN",
                "plate_confidence": 0.0,
                "plate_crop_path": "",
                "first_frame": record.get("first_frame", 0),
                "last_frame": record.get("last_frame", 0),
                "crop_paths": [],
                "best_crop_path": record.get("best_crop_path", ""),
                "best_confidence": record.get("best_confidence", 0.0),
            }
            groups[canonical_plate] = group
        return group

    def add_record_to_group(group: Dict[str, Any], record: Dict[str, Any], crop_paths: List[str],
                            plate_confidence: float, plate_crop_path: str) -> None:
        group["track_ids"].extend(record.get("track_ids", []))
        group["first_frame"] = min(group["first_frame"], record.get("first_frame", group["first_frame"]))
        group["last_frame"] = max(group["last_frame"], record.get("last_frame", group["last_frame"]))
        group["crop_paths"].extend(crop_paths)

        if plate_confidence>group["plate_confidence"]:
            group["plate_confidence"] = plate_confidence
            group["plate_crop_path"] = plate_crop_path

        if record.get("best_confidence", 0.0)>group.get("best_confidence", 0.0):
            group["best_confidence"] = record.get("best_confidence", 0.0)
            group["best_crop_path"] = record.get("best_crop_path", "")

    def merge_observation_plate_variants(canonical_observations: List[tuple]) -> List[tuple]:
        if not merge_near_duplicate_plate_text or len(canonical_observations)<2:
            return canonical_observations

        score_by_plate: Dict[str, float] = {}
        count_by_plate: Dict[str, int] = {}
        for canonical_plate, observation in canonical_observations:
            score_by_plate[canonical_plate] = score_by_plate.get(canonical_plate, 0.0) + float(observation.get("plate_confidence", 0.0))
            count_by_plate[canonical_plate] = count_by_plate.get(canonical_plate, 0) + 1

        ranked_plates = sorted(
            score_by_plate,
            key=lambda plate: (score_by_plate[plate], count_by_plate[plate]),
            reverse=True,
        )
        alias_map = {}
        for source_plate in ranked_plates:
            for target_plate in ranked_plates:
                if source_plate==target_plate:
                    continue
                target_is_stronger = (
                    count_by_plate[target_plate]>count_by_plate[source_plate]
                    or score_by_plate[target_plate]>=score_by_plate[source_plate]
                )
                if target_is_stronger and is_near_duplicate_plate(source_plate, target_plate, near_duplicate_max_distance):
                    alias_map[source_plate] = target_plate
                    break

        return [(alias_map.get(canonical_plate, canonical_plate), observation) for canonical_plate, observation in canonical_observations]

    def merge_group_into(target: Dict[str, Any], source: Dict[str, Any]) -> None:
        target["track_ids"].extend(source.get("track_ids", []))
        target["first_frame"] = min(target["first_frame"], source.get("first_frame", target["first_frame"]))
        target["last_frame"] = max(target["last_frame"], source.get("last_frame", target["last_frame"]))
        target["crop_paths"].extend(source.get("crop_paths", []))

        if source.get("plate_confidence", 0.0)>target.get("plate_confidence", 0.0):
            target["plate_confidence"] = source.get("plate_confidence", 0.0)
            target["plate_crop_path"] = source.get("plate_crop_path", "")

        if source.get("best_confidence", 0.0)>target.get("best_confidence", 0.0):
            target["best_confidence"] = source.get("best_confidence", 0.0)
            target["best_crop_path"] = source.get("best_crop_path", "")

    def parse_crop_frame(crop_path: str) -> Optional[int]:
        match = re.search(r"_frame_(\d+)", str(crop_path))
        if not match:
            return None
        return int(match.group(1))

    def get_manual_track_override(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for track_id in record.get("track_ids", []):
            if int(track_id) in manual_track_plate_overrides:
                return manual_track_plate_overrides[int(track_id)]
        return None

    def filter_manual_override_crop_paths(crop_paths: List[str], override: Dict[str, Any]) -> List[str]:
        start_frame = override.get("start_frame")
        end_frame = override.get("end_frame")
        if start_frame is None and end_frame is None:
            return crop_paths

        selected = []
        for crop_path in crop_paths:
            crop_frame = parse_crop_frame(crop_path)
            if crop_frame is None:
                continue
            if start_frame is not None and crop_frame<int(start_frame):
                continue
            if end_frame is not None and crop_frame>int(end_frame):
                continue
            selected.append(crop_path)
        return selected

    def get_observation_crop_window(record: Dict[str, Any], source_crop_path: str, canonical_plate: str,
                                    observed_plate_by_crop: Dict[str, str]) -> List[str]:
        if not source_crop_path:
            return []

        crop_paths = get_record_crop_paths(record)
        if source_crop_path not in crop_paths or observation_neighbor_crops<=0:
            return [source_crop_path]

        source_index = crop_paths.index(source_crop_path)
        start_index = max(0, source_index-observation_neighbor_crops)
        end_index = min(len(crop_paths), source_index+observation_neighbor_crops+1)
        selected = []
        observed_indices = [
            (crop_paths.index(crop_path), observed_plate)
            for crop_path, observed_plate in observed_plate_by_crop.items()
            if crop_path in crop_paths
        ]
        for crop_path in crop_paths[start_index:end_index]:
            crop_index = crop_paths.index(crop_path)
            observed_plate = observed_plate_by_crop.get(crop_path)
            if observed_plate is not None and observed_plate!=canonical_plate:
                continue
            if observed_plate is None and observed_indices:
                nearest_distance = min(abs(crop_index-observed_index) for observed_index, _ in observed_indices)
                nearest_plates = {
                    observed_plate
                    for observed_index, observed_plate in observed_indices
                    if abs(crop_index-observed_index)==nearest_distance
                }
                if nearest_plates!={canonical_plate}:
                    continue
            selected.append(crop_path)
        return selected

    for record in records:
        record_crop_paths = get_record_crop_paths(record)
        if len(record_crop_paths)<min_crop_count:
            continue

        manual_override = get_manual_track_override(record)
        if manual_override and manual_override.get("plate"):
            canonical_plate = canonicalize_plate(
                manual_override["plate"],
                expected_plates,
                max_edit_distance,
                default_province,
                default_city_letter,
                plate_aliases,
                allow_unmatched_legal,
            )
            manual_crop_paths = filter_manual_override_crop_paths(record_crop_paths, manual_override)
            if canonical_plate and manual_crop_paths:
                group = get_group(canonical_plate, record)
                add_record_to_group(
                    group,
                    record,
                    manual_crop_paths,
                    float(record.get("plate_confidence", 0.0)),
                    record.get("plate_crop_path", ""),
                )
                continue

        observations = record.get("plate_observations", [])
        canonical_observations = []
        for observation in observations:
            canonical_plate = canonicalize_plate(
                observation.get("plate_text", "UNKNOWN"),
                expected_plates,
                max_edit_distance,
                default_province,
                default_city_letter,
                plate_aliases,
                allow_unmatched_legal,
            )
            if canonical_plate is None:
                continue
            canonical_observations.append((canonical_plate, observation))
        canonical_observations = merge_observation_plate_variants(canonical_observations)
        all_observed_plate_by_crop = {
            observation.get("source_crop_path", ""): canonical_plate
            for canonical_plate, observation in canonical_observations
            if observation.get("source_crop_path", "")
        }
        had_conflicting_observations = len(set(item[0] for item in canonical_observations))>1

        if had_conflicting_observations and drop_minor_conflicting_observations:
            score_by_plate: Dict[str, float] = {}
            count_by_plate: Dict[str, int] = {}
            for canonical_plate, observation in canonical_observations:
                score_by_plate[canonical_plate] = score_by_plate.get(canonical_plate, 0.0) + float(observation.get("plate_confidence", 0.0))
                count_by_plate[canonical_plate] = count_by_plate.get(canonical_plate, 0) + 1

            ranked_plates = sorted(
                score_by_plate,
                key=lambda plate: (score_by_plate[plate], count_by_plate[plate]),
                reverse=True,
            )
            dominant_plate = ranked_plates[0] if ranked_plates else ""
            second_score = score_by_plate[ranked_plates[1]] if len(ranked_plates)>1 else 0.0
            dominant_is_clear = (
                dominant_plate
                and count_by_plate[dominant_plate]>=dominant_plate_min_votes
                and score_by_plate[dominant_plate]>=max(second_score*dominant_plate_score_ratio, second_score+1e-6)
            )
            if dominant_is_clear:
                canonical_observations = [
                    item for item in canonical_observations
                    if item[0]==dominant_plate
                ]

        unique_observation_plates = sorted(set(item[0] for item in canonical_observations))
        if len(unique_observation_plates)==1:
            canonical_plate = unique_observation_plates[0]
            group = get_group(canonical_plate, record)
            best_observation = max(canonical_observations, key=lambda item: item[1].get("plate_confidence", 0.0))[1]
            if had_conflicting_observations:
                crop_paths = []
                for _, observation in canonical_observations:
                    crop_paths.extend(get_observation_crop_window(
                        record,
                        observation.get("source_crop_path", ""),
                        canonical_plate,
                        all_observed_plate_by_crop,
                    ))
                crop_paths = sorted(set(crop_paths))
            else:
                crop_paths = record_crop_paths
            add_record_to_group(
                group,
                record,
                crop_paths,
                float(best_observation.get("plate_confidence", 0.0)),
                best_observation.get("plate_crop_path", ""),
            )
            continue

        if len(unique_observation_plates)>1:
            score_by_plate: Dict[str, float] = {}
            count_by_plate: Dict[str, int] = {}
            best_observation_by_plate: Dict[str, Dict[str, Any]] = {}
            observed_crop_paths = set()
            observed_plate_by_crop = {
                observation.get("source_crop_path", ""): canonical_plate
                for canonical_plate, observation in canonical_observations
                if observation.get("source_crop_path", "")
            }

            for canonical_plate, observation in canonical_observations:
                source_crop_path = observation.get("source_crop_path", "")
                if source_crop_path:
                    observed_crop_paths.add(source_crop_path)
                score_by_plate[canonical_plate] = score_by_plate.get(canonical_plate, 0.0) + float(observation.get("plate_confidence", 0.0))
                count_by_plate[canonical_plate] = count_by_plate.get(canonical_plate, 0) + 1
                if (
                    canonical_plate not in best_observation_by_plate
                    or observation.get("plate_confidence", 0.0)>best_observation_by_plate[canonical_plate].get("plate_confidence", 0.0)
                ):
                    best_observation_by_plate[canonical_plate] = observation

                crop_paths = get_observation_crop_window(record, source_crop_path, canonical_plate, observed_plate_by_crop)
                if not crop_paths:
                    continue
                group = get_group(canonical_plate, record)
                add_record_to_group(
                    group,
                    record,
                    crop_paths,
                    float(observation.get("plate_confidence", 0.0)),
                    observation.get("plate_crop_path", ""),
                )

            if assign_unobserved_crops and score_by_plate:
                ranked_plates = sorted(
                    score_by_plate,
                    key=lambda plate: (score_by_plate[plate], count_by_plate[plate]),
                    reverse=True,
                )
                dominant_plate = ranked_plates[0]
                second_score = score_by_plate[ranked_plates[1]] if len(ranked_plates)>1 else 0.0
                dominant_is_clear = (
                    count_by_plate[dominant_plate]>=dominant_plate_min_votes
                    and score_by_plate[dominant_plate]>=max(second_score*dominant_plate_score_ratio, second_score+1e-6)
                )
                if dominant_is_clear:
                    unobserved_crop_paths = [
                        crop_path for crop_path in record_crop_paths
                        if crop_path not in observed_crop_paths
                    ]
                    if unobserved_crop_paths:
                        best_observation = best_observation_by_plate[dominant_plate]
                        group = get_group(dominant_plate, record)
                        add_record_to_group(
                            group,
                            record,
                            unobserved_crop_paths,
                            float(best_observation.get("plate_confidence", 0.0)),
                            best_observation.get("plate_crop_path", ""),
                        )
            continue

        canonical_plate = canonicalize_plate(
            record.get("plate_text", "UNKNOWN"),
            expected_plates,
            max_edit_distance,
            default_province,
            default_city_letter,
            plate_aliases,
            allow_unmatched_legal,
        )
        if canonical_plate is None:
            if not keep_unknown:
                continue
            canonical_plate = f"UNKNOWN_{unknown_index:04d}"
            unknown_index += 1
        group = get_group(canonical_plate, record)
        add_record_to_group(
            group,
            record,
            record_crop_paths,
            float(record.get("plate_confidence", 0.0)),
            record.get("plate_crop_path", ""),
        )

    if merge_near_duplicate_plate_text:
        changed = True
        while changed:
            changed = False
            keys = list(groups.keys())
            for index, left_key in enumerate(keys):
                if left_key not in groups:
                    continue
                for right_key in keys[index+1:]:
                    if right_key not in groups:
                        continue
                    if not is_near_duplicate_plate(left_key, right_key, near_duplicate_max_distance):
                        continue

                    left_group = groups[left_key]
                    right_group = groups[right_key]
                    left_crop_count = len(left_group.get("crop_paths", []))
                    right_crop_count = len(right_group.get("crop_paths", []))
                    track_overlap = bool(set(left_group.get("track_ids", [])) & set(right_group.get("track_ids", [])))
                    has_small_variant = (
                        left_crop_count<=near_duplicate_small_group_max_crops
                        or right_crop_count<=near_duplicate_small_group_max_crops
                    )
                    if not track_overlap and not has_small_variant:
                        continue

                    left_rank = (left_crop_count, left_group.get("plate_confidence", 0.0))
                    right_rank = (right_crop_count, right_group.get("plate_confidence", 0.0))
                    if left_rank>=right_rank:
                        merge_group_into(left_group, right_group)
                        del groups[right_key]
                    else:
                        merge_group_into(right_group, left_group)
                        del groups[left_key]
                    changed = True
                    break
                if changed:
                    break

    consolidated = []
    for group in groups.values():
        group["track_ids"] = sorted(set(group["track_ids"]))
        group["crop_paths"] = sorted(set(group["crop_paths"]))
        consolidated.append(group)

    expected_order = {plate: index for index, plate in enumerate(normalize_expected_plates(expected_plates))}
    consolidated.sort(key=lambda item: (expected_order.get(item["vehicle_id"], 999), item["first_frame"]))
    return consolidated
