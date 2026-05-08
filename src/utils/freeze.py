def apply_freeze(model, freeze_modules: list[str] | None):
    """Set requires_grad=False for specified module paths.

    Args:
        model: The top-level model (LatentSeekerForConditionalGeneration).
        freeze_modules: e.g. ["model.language_model", "model.longtext.layers"].
    """
    if not freeze_modules:
        return

    for module_path in freeze_modules:
        module = model
        for attr in module_path.split("."):
            module = getattr(module, attr)
        for param in module.parameters():
            param.requires_grad = False
