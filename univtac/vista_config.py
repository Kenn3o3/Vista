from __future__ import annotations


CANONICAL_VISTA_CONFIG = "so3"
VISTA_SO2_CONFIG = "so2"
VISTA_SO3_GLOBAL_L0_CONFIG = "so3_global_l0"
VISTA_SO2_GLOBAL_L0_CONFIG = "so2_global_l0"
VISTA_SO3_EARLY_FUSION_CONFIG = "so3_early_fusion"
VISTA_SO2_EARLY_FUSION_CONFIG = "so2_early_fusion"
VISTA_SO3_CONCATE_CONFIG = "so3_concate"
VISTA_SO2_CONCATE_CONFIG = "so2_concate"
SUPPORTED_VISTA_CONFIGS = (
    CANONICAL_VISTA_CONFIG,
    VISTA_SO2_CONFIG,
    VISTA_SO3_GLOBAL_L0_CONFIG,
    VISTA_SO2_GLOBAL_L0_CONFIG,
    VISTA_SO3_EARLY_FUSION_CONFIG,
    VISTA_SO2_EARLY_FUSION_CONFIG,
    VISTA_SO3_CONCATE_CONFIG,
    VISTA_SO2_CONCATE_CONFIG,
)

_VISTA_CONFIG_ALIASES = {
    "vista": CANONICAL_VISTA_CONFIG,
    "so3": CANONICAL_VISTA_CONFIG,
    "vista_so3": CANONICAL_VISTA_CONFIG,
    "current": CANONICAL_VISTA_CONFIG,
    "so2": VISTA_SO2_CONFIG,
    "vista_so2": VISTA_SO2_CONFIG,
    "so3_global_l0": VISTA_SO3_GLOBAL_L0_CONFIG,
    "vista_so3_global_l0": VISTA_SO3_GLOBAL_L0_CONFIG,
    "global_l0": VISTA_SO3_GLOBAL_L0_CONFIG,
    "vista_global_l0": VISTA_SO3_GLOBAL_L0_CONFIG,
    "so2_global_l0": VISTA_SO2_GLOBAL_L0_CONFIG,
    "vista_so2_global_l0": VISTA_SO2_GLOBAL_L0_CONFIG,
    "so3_early_fusion": VISTA_SO3_EARLY_FUSION_CONFIG,
    "vista_so3_early_fusion": VISTA_SO3_EARLY_FUSION_CONFIG,
    "early_fusion": VISTA_SO3_EARLY_FUSION_CONFIG,
    "vista_early_fusion": VISTA_SO3_EARLY_FUSION_CONFIG,
    "so2_early_fusion": VISTA_SO2_EARLY_FUSION_CONFIG,
    "vista_so2_early_fusion": VISTA_SO2_EARLY_FUSION_CONFIG,
    "so3_concate": VISTA_SO3_CONCATE_CONFIG,
    "vista_so3_concate": VISTA_SO3_CONCATE_CONFIG,
    "concate": VISTA_SO3_CONCATE_CONFIG,
    "vista_concate": VISTA_SO3_CONCATE_CONFIG,
    "so3_concat": VISTA_SO3_CONCATE_CONFIG,
    "vista_so3_concat": VISTA_SO3_CONCATE_CONFIG,
    "concat": VISTA_SO3_CONCATE_CONFIG,
    "vista_concat": VISTA_SO3_CONCATE_CONFIG,
    "so2_concate": VISTA_SO2_CONCATE_CONFIG,
    "vista_so2_concate": VISTA_SO2_CONCATE_CONFIG,
    "so2_concat": VISTA_SO2_CONCATE_CONFIG,
    "vista_so2_concat": VISTA_SO2_CONCATE_CONFIG,
}

_REMOVED_VISTA_CONFIGS = {
    "marker_flow",
    "flow",
    "normal",
    "flow_normal",
    "normal_flow",
    "marker_flow_normal",
    "vista_marker_flow",
    "vista_normal",
    "vista_flow_normal",
}


def normalize_vista_config_name(config_name: str | None) -> str:
    if config_name is None:
        raise ValueError(
            "VISTA config is required. Use 'vista' for wrist camera + tactile marker RGB."
        )

    value = str(config_name).strip()
    if value in _VISTA_CONFIG_ALIASES:
        return _VISTA_CONFIG_ALIASES[value]

    if value in _REMOVED_VISTA_CONFIGS:
        raise ValueError(
            f"VISTA config '{value}' has been removed. "
            "Supported VISTA configs are 'so3', 'so2', and registered fusion ablations."
        )

    raise ValueError(
        f"Unsupported VISTA config '{value}'. "
        "Supported VISTA configs are 'so3', 'so2', "
        "'so3_global_l0', 'so2_global_l0', "
        "'so3_early_fusion', 'so2_early_fusion', "
        "'so3_concate', and 'so2_concate'."
    )
