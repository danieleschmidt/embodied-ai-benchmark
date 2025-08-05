"""Internationalization and localization utilities for the embodied AI benchmark."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os
import locale
from datetime import datetime
import gettext

logger = logging.getLogger(__name__)


class LocalizationManager:
    """Manages internationalization and localization for the benchmark."""
    
    def __init__(self, locale_dir: Optional[str] = None, default_locale: str = "en_US"):
        """Initialize localization manager.
        
        Args:
            locale_dir: Directory containing locale files
            default_locale: Default locale to use
        """
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.locale_dir = Path(locale_dir) if locale_dir else Path(__file__).parent.parent / "locales"
        
        self._translations: Dict[str, Dict[str, str]] = {}
        self._load_translations()
        
        # Set system locale
        try:
            locale.setlocale(locale.LC_ALL, self.current_locale)
        except locale.Error:
            logger.warning(f"Failed to set locale to {self.current_locale}, using default")
    
    def _load_translations(self):
        """Load translation files from locale directory."""
        if not self.locale_dir.exists():
            logger.warning(f"Locale directory not found: {self.locale_dir}")
            return
        
        for locale_file in self.locale_dir.glob("*.json"):
            locale_code = locale_file.stem
            try:
                with open(locale_file, 'r', encoding='utf-8') as f:
                    self._translations[locale_code] = json.load(f)
                logger.info(f"Loaded translations for locale: {locale_code}")
            except Exception as e:
                logger.error(f"Failed to load translations for {locale_code}: {e}")
    
    def set_locale(self, locale_code: str) -> bool:
        """Set the current locale.
        
        Args:
            locale_code: Locale code (e.g., 'en_US', 'es_ES', 'zh_CN')
            
        Returns:
            True if locale was set successfully
        """
        if locale_code in self._translations or locale_code == self.default_locale:
            self.current_locale = locale_code
            
            # Update system locale
            try:
                locale.setlocale(locale.LC_ALL, locale_code)
                logger.info(f"Locale set to: {locale_code}")
                return True
            except locale.Error:
                logger.warning(f"System locale {locale_code} not available, using translations only")
                return True
        else:
            logger.error(f"Translations not available for locale: {locale_code}")
            return False
    
    def get_available_locales(self) -> Dict[str, str]:
        """Get available locales with their display names.
        
        Returns:
            Dictionary mapping locale codes to display names
        """
        locales = {self.default_locale: "English (US)"}
        
        for locale_code in self._translations.keys():
            # Get locale display name from translations or use code
            display_name = self._translations[locale_code].get("_locale_name", locale_code)
            locales[locale_code] = display_name
        
        return locales
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate a message key to the current locale.
        
        Args:
            key: Translation key
            **kwargs: Variables to substitute in the translation
            
        Returns:
            Translated message
        """
        # Get translation for current locale
        if self.current_locale in self._translations:
            message = self._translations[self.current_locale].get(key)
            if message:
                try:
                    return message.format(**kwargs)
                except KeyError as e:
                    logger.warning(f"Missing variable {e} in translation for key: {key}")
                    return message
        
        # Fallback to default locale
        if self.default_locale in self._translations:
            message = self._translations[self.default_locale].get(key)
            if message:
                try:
                    return message.format(**kwargs)
                except KeyError:
                    return message
        
        # Final fallback to key itself
        logger.warning(f"Translation not found for key: {key}")
        return key
    
    def format_datetime(self, dt: datetime, format_type: str = "medium") -> str:
        """Format datetime according to current locale.
        
        Args:
            dt: Datetime to format
            format_type: Format type ('short', 'medium', 'long', 'full')
            
        Returns:
            Formatted datetime string
        """
        try:
            if format_type == "short":
                return dt.strftime("%x %X")  # Locale-specific short format
            elif format_type == "medium":
                return dt.strftime("%c")     # Locale-specific medium format
            elif format_type == "long":
                return dt.strftime("%A, %B %d, %Y at %I:%M:%S %p")
            else:  # full
                return dt.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
        except:
            # Fallback to ISO format
            return dt.isoformat()
    
    def format_number(self, number: Union[int, float], decimal_places: Optional[int] = None) -> str:
        """Format number according to current locale.
        
        Args:
            number: Number to format
            decimal_places: Number of decimal places
            
        Returns:
            Formatted number string
        """
        try:
            if decimal_places is not None:
                return locale.format_string(f"%.{decimal_places}f", number, grouping=True)
            elif isinstance(number, int):
                return locale.format_string("%d", number, grouping=True)
            else:
                return locale.format_string("%.2f", number, grouping=True)
        except:
            # Fallback to string conversion
            return str(number)
    
    def format_currency(self, amount: float, currency_code: str = "USD") -> str:
        """Format currency according to current locale.
        
        Args:
            amount: Amount to format
            currency_code: Currency code (e.g., 'USD', 'EUR', 'JPY')
            
        Returns:
            Formatted currency string
        """
        try:
            # Simple currency formatting
            symbol_map = {
                "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥",
                "CNY": "¥", "INR": "₹", "CAD": "C$", "AUD": "A$"
            }
            symbol = symbol_map.get(currency_code, currency_code)
            formatted_amount = self.format_number(amount, 2)
            return f"{symbol}{formatted_amount}"
        except:
            return f"{amount} {currency_code}"
    
    def get_text_direction(self) -> str:
        """Get text direction for current locale.
        
        Returns:
            'ltr' for left-to-right, 'rtl' for right-to-left
        """
        rtl_locales = ['ar', 'he', 'fa', 'ur', 'yi']
        lang_code = self.current_locale.split('_')[0]
        return 'rtl' if lang_code in rtl_locales else 'ltr'


class MessageCatalog:
    """Manages error messages and user-facing text with internationalization."""
    
    def __init__(self, localization_manager: LocalizationManager):
        """Initialize message catalog.
        
        Args:
            localization_manager: Localization manager instance
        """
        self.i18n = localization_manager
    
    # Error Messages
    def validation_error(self, field: str, value: Any) -> str:
        """Get validation error message."""
        return self.i18n.translate("validation_error", field=field, value=value)
    
    def file_not_found_error(self, filename: str) -> str:
        """Get file not found error message."""
        return self.i18n.translate("file_not_found", filename=filename)
    
    def permission_denied_error(self, action: str) -> str:
        """Get permission denied error message."""
        return self.i18n.translate("permission_denied", action=action)
    
    def timeout_error(self, timeout_seconds: int) -> str:
        """Get timeout error message."""
        return self.i18n.translate("timeout_error", timeout=timeout_seconds)
    
    # Success Messages
    def task_completed(self, task_name: str, duration: float) -> str:
        """Get task completion message."""
        formatted_duration = self.i18n.format_number(duration, 2)
        return self.i18n.translate("task_completed", task=task_name, duration=formatted_duration)
    
    def benchmark_finished(self, num_episodes: int, success_rate: float) -> str:
        """Get benchmark completion message."""
        formatted_rate = self.i18n.format_number(success_rate * 100, 1)
        return self.i18n.translate("benchmark_finished", episodes=num_episodes, success_rate=formatted_rate)
    
    # Progress Messages
    def episode_progress(self, current: int, total: int) -> str:
        """Get episode progress message."""
        return self.i18n.translate("episode_progress", current=current, total=total)
    
    def loading_model(self, model_name: str) -> str:
        """Get model loading message."""
        return self.i18n.translate("loading_model", model=model_name)
    
    # Status Messages
    def agent_performance(self, agent_name: str, score: float) -> str:
        """Get agent performance message."""
        formatted_score = self.i18n.format_number(score, 3)
        return self.i18n.translate("agent_performance", agent=agent_name, score=formatted_score)
    
    def system_status(self, cpu_usage: float, memory_usage: float) -> str:
        """Get system status message."""
        cpu_pct = self.i18n.format_number(cpu_usage, 1)
        mem_pct = self.i18n.format_number(memory_usage, 1)
        return self.i18n.translate("system_status", cpu=cpu_pct, memory=mem_pct)


# Global instances
default_i18n = LocalizationManager()
messages = MessageCatalog(default_i18n)


def init_i18n(locale_dir: Optional[str] = None, locale: str = "en_US") -> LocalizationManager:
    """Initialize internationalization system.
    
    Args:
        locale_dir: Directory containing locale files
        locale: Default locale to use
        
    Returns:
        Configured LocalizationManager instance
    """
    global default_i18n, messages
    
    default_i18n = LocalizationManager(locale_dir, locale)
    messages = MessageCatalog(default_i18n)
    
    # Auto-detect system locale
    try:
        system_locale = locale.getdefaultlocale()[0]
        if system_locale and system_locale in default_i18n.get_available_locales():
            default_i18n.set_locale(system_locale)
            logger.info(f"Auto-detected system locale: {system_locale}")
    except:
        logger.info("Using default locale")
    
    return default_i18n


def t(key: str, **kwargs) -> str:
    """Shorthand translation function.
    
    Args:
        key: Translation key
        **kwargs: Variables to substitute
        
    Returns:
        Translated message
    """
    return default_i18n.translate(key, **kwargs)


def set_locale(locale_code: str) -> bool:
    """Set global locale.
    
    Args:
        locale_code: Locale code to set
        
    Returns:
        True if successful
    """
    return default_i18n.set_locale(locale_code)


def get_available_locales() -> Dict[str, str]:
    """Get available locales.
    
    Returns:
        Dictionary of locale codes to display names
    """
    return default_i18n.get_available_locales()