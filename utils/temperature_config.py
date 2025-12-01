"""Temperature configuration utilities for different tasks and providers."""

from typing import Optional, Dict, Any


def get_temperature_for_task(
    config: dict,
    task_type: str = "default",
    provider: Optional[str] = None,
    fallback_temperature: float = 0.3
) -> float:
    """Get temperature configuration for a specific task and provider.
    
    Args:
        config: Configuration dictionary
        task_type: Type of task (default, direct_answer, search_query, etc.)
        provider: LLM provider name (optional)
        fallback_temperature: Default temperature if no configuration found
        
    Returns:
        Temperature value for the task/provider
    """
    temp_settings = config.get("temperature_settings", {})
    
    # Check provider-specific task temperature first
    if provider:
        provider_config = temp_settings.get("providers", {}).get(provider, {})
        provider_task_temp = provider_config.get("tasks", {}).get(task_type)
        if provider_task_temp is not None:
            return float(provider_task_temp)
        
        # Check provider default temperature
        provider_default = provider_config.get("default")
        if provider_default is not None:
            return float(provider_default)
    
    # Check global task temperature
    task_temp = temp_settings.get("tasks", {}).get(task_type)
    if task_temp is not None:
        return float(task_temp)
    
    # Fall back to global default
    global_default = temp_settings.get("default")
    if global_default is not None:
        return float(global_default)
    
    # Final fallback
    return float(fallback_temperature)