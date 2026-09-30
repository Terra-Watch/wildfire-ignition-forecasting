from wildfire.reporting import labels, models, style


def test_arm_names_map_to_one_colour():
    assert style.base_arm("boosting@enriched") == "boosting" == style.base_arm("boosting#3")
    assert style.colour("boosting@tabular") == style.colour("boosting@enriched")
    assert style.colour("unknown_arm") == style.COLOURS["neutral"]


def test_rule_table_reads_the_configuration():
    t = labels.rule_table()
    assert "0.2" in t[t.rule == "forest domain"].setting.iloc[0] and len(t) == 6


def test_ladder_table_lists_every_arm_with_its_inputs():
    t = models.ladder_table()
    assert t.arm.tolist() == ["climatology_cell", "landcover_season", "fwi_season", "logistic_weather_static",
                              "logistic_with_burn", "boosting", "tabular_network"]
    assert t.set_index("arm").loc["boosting", "tabular_inputs"] == 42 and t.set_index("arm").loc["tabular_network", "levels"] == "tabular, enriched"


def test_diagnostics_are_the_three_declared_arms():
    assert models.diagnostics_table().arm.tolist() == ["coords_only", "cell_id_only", "static_only"]
