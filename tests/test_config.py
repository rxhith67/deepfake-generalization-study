from src.config import deep_merge, load_config


def test_deep_merge_preserves_nested_values():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {
        "a": {"b": 3, "c": 2}
    }


def test_model_config_inherits_base():
    config = load_config("config/xception.yaml")
    assert config["model"]["name"] == "xception"
    # Xception overrides the base resolution with its pretrained recipe.
    assert config["training"]["image_size"] == 299
    assert config["split"]["by"] == "video_id"
