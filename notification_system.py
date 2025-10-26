#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔔 Система уведомлений через Telegram бота
"""

import requests
import json
import asyncio
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from time_monitor import TimeTrigger, TriggerType, UserSettings

@dataclass
class NotificationMessage:
    """Сообщение уведомления"""
    chat_id: int
    text: str
    parse_mode: str = "HTML"
    reply_markup: Optional[Dict] = None
    priority: int = 1

class TelegramNotifier:
    """Отправка уведомлений через Telegram"""

    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        self.session = requests.Session()
        self.session.verify = False  # Для решения SSL проблем

    def send_message(self, message: NotificationMessage) -> bool:
        """Отправляет сообщение в Telegram"""
        try:
            url = f"{self.api_url}/sendMessage"
            data = {
                "chat_id": message.chat_id,
                "text": message.text,
                "parse_mode": message.parse_mode
            }

            if message.reply_markup:
                data["reply_markup"] = json.dumps(message.reply_markup)

            response = self.session.post(url, json=data, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if result.get("ok"):
                    print(f"✅ Уведомление отправлено пользователю {message.chat_id}")
                    return True
                else:
                    print(f"❌ Ошибка Telegram API: {result.get('description')}")
                    return False
            else:
                print(f"❌ HTTP ошибка: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Ошибка отправки уведомления: {e}")
            return False

    def send_departure_reminder(self, chat_id: int, client_info: Dict, departure_time: datetime) -> bool:
        """Отправляет напоминание о выезде"""
        text = f"""
⏰ <b>Напоминание о выезде</b>

🚗 Через 15 мин выезжайте к клиенту
📍 <b>Адрес:</b> {client_info.get('address', 'Не указан')}
👤 <b>Клиент:</b> {client_info.get('client_level', 'Стандарт')}
⏱️ <b>Время в пути:</b> {client_info.get('travel_time', 'Не указано')} мин
🕐 <b>Выезд в:</b> {departure_time.strftime('%H:%M')}
        """.strip()

        message = NotificationMessage(
            chat_id=chat_id,
            text=text,
            priority=2
        )

        return self.send_message(message)

    def send_lunch_reminder(self, chat_id: int, lunch_start: datetime, lunch_end: datetime) -> bool:
        """Отправляет напоминание об обеде"""
        text = f"""
🍽️ <b>Время обеда</b>

⏰ Время обеда через 15 минут
🕐 <b>С:</b> {lunch_start.strftime('%H:%M')}
🕐 <b>До:</b> {lunch_end.strftime('%H:%M')}
⏱️ <b>Продолжительность:</b> 1 час

💡 <i>Не забудьте учесть время обеда при планировании маршрута</i>
        """.strip()

        message = NotificationMessage(
            chat_id=chat_id,
            text=text,
            priority=2
        )

        return self.send_message(message)

    def send_delay_alert(self, chat_id: int, client_info: Dict, delay_minutes: int, planned_arrival: datetime) -> bool:
        """Отправляет уведомление об опоздании"""
        text = f"""
🚨 <b>ОПОЗДАНИЕ!</b>

⚠️ Вы опаздываете на <b>{delay_minutes} мин</b> к клиенту
📍 <b>Адрес:</b> {client_info.get('address', 'Не указан')}
👤 <b>Клиент:</b> {client_info.get('client_level', 'Стандарт')}
🕐 <b>Планируемое прибытие:</b> {planned_arrival.strftime('%H:%M')}
🕐 <b>Текущее время:</b> {datetime.now().strftime('%H:%M')}

💡 <i>Рекомендуется связаться с клиентом и предупредить об опоздании</i>
        """.strip()

        message = NotificationMessage(
            chat_id=chat_id,
            text=text,
            priority=1
        )

        return self.send_message(message)

    def send_traffic_alert(self, chat_id: int, route_info: Dict, time_increase: int, old_time: int, new_time: int) -> bool:
        """Отправляет уведомление об изменении трафика"""
        text = f"""
🚗 <b>Изменение трафика</b>

⚠️ Пробка на маршруте!
⏱️ <b>Время увеличено на:</b> {time_increase} мин
🕐 <b>Было:</b> {old_time} мин
🕐 <b>Стало:</b> {new_time} мин
📍 <b>Маршрут:</b> {route_info.get('description', 'Не указан')}

💡 <i>Маршрут будет автоматически пересчитан</i>
        """.strip()

        message = NotificationMessage(
            chat_id=chat_id,
            text=text,
            priority=1
        )

        return self.send_message(message)

    def send_route_update(self, chat_id: int, route_info: Dict) -> bool:
        """Отправляет уведомление об обновлении маршрута"""
        text = f"""
🔄 <b>Маршрут обновлен</b>

📍 <b>Новое местоположение:</b> {route_info.get('user_location', 'Не указано')}
📊 <b>Клиентов в маршруте:</b> {route_info.get('total_clients', 0)}
⏱️ <b>Общее время:</b> {route_info.get('total_time', 'Не указано')}
🛣️ <b>Общее расстояние:</b> {route_info.get('total_distance', 'Не указано')}

💡 <i>Маршрут пересчитан с учетом нового местоположения</i>
        """.strip()

        message = NotificationMessage(
            chat_id=chat_id,
            text=text,
            priority=2
        )

        return self.send_message(message)

    def send_client_arrival(self, chat_id: int, client_info: Dict, arrival_time: datetime) -> bool:
        """Отправляет уведомление о прибытии к клиенту"""
        service_time = client_info.get('service_time', 20)
        client_level = client_info.get('client_level', 'Стандарт')

        text = f"""
📍 <b>Прибытие к клиенту</b>

✅ Вы прибыли к клиенту
📍 <b>Адрес:</b> {client_info.get('address', 'Не указан')}
👤 <b>Клиент:</b> {client_level}
⏱️ <b>Время обслуживания:</b> {service_time} мин
🕐 <b>Прибытие:</b> {arrival_time.strftime('%H:%M')}
🕐 <b>Окончание:</b> {(arrival_time + timedelta(minutes=service_time)).strftime('%H:%M')}

💡 <i>Не забудьте отметить клиента как посещенного после обслуживания</i>
        """.strip()

        # Добавляем кнопку для отметки клиента
        reply_markup = {
            "inline_keyboard": [[
                {
                    "text": f"✅ Отметить как посещенного",
                    "callback_data": f"visited_{client_info.get('id', 0)}"
                }
            ]]
        }

        message = NotificationMessage(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
            priority=2
        )

        return self.send_message(message)

    def send_settings_menu(self, chat_id: int, settings: UserSettings) -> bool:
        """Отправляет меню настроек"""
        status_emoji = "✅" if settings.notifications_enabled else "❌"

        text = f"""
⚙️ <b>Настройки уведомлений</b>

🔔 <b>Уведомления:</b> {status_emoji} {'Включены' if settings.notifications_enabled else 'Выключены'}
⏰ <b>Напоминания о выезде:</b> за {settings.departure_reminder_minutes} мин
🍽️ <b>Напоминания об обеде:</b> за {settings.lunch_reminder_minutes} мин
🚨 <b>Опоздания:</b> от {settings.delay_threshold_minutes} мин
🚗 <b>Трафик:</b> от {settings.traffic_threshold_minutes} мин
🕐 <b>Рабочее время:</b> {settings.work_start}-{settings.work_end}
🍽️ <b>Обеденное время:</b> {settings.lunch_start}-{settings.lunch_end}
🌍 <b>Язык:</b> {settings.language}
        """.strip()

        reply_markup = {
            "inline_keyboard": [
                [
                    {"text": "🔔 Уведомления", "callback_data": "settings_notifications"},
                    {"text": "⏰ Напоминания", "callback_data": "settings_reminders"}
                ],
                [
                    {"text": "🚗 Трафик", "callback_data": "settings_traffic"},
                    {"text": "🕐 Время", "callback_data": "settings_time"}
                ],
                [
                    {"text": "🌍 Язык", "callback_data": "settings_language"},
                    {"text": "❌ Закрыть", "callback_data": "settings_close"}
                ]
            ]
        }

        message = NotificationMessage(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
            priority=3
        )

        return self.send_message(message)

    def send_help_message(self, chat_id: int) -> bool:
        """Отправляет справку по командам"""
        text = """
🤖 <b>Команды бота</b>

<b>Основные команды:</b>
/start - Регистрация и настройка
/help - Эта справка
/status - Текущий статус маршрута
/settings - Настройки уведомлений

<b>Управление маршрутом:</b>
/next - Следующий клиент
/visited <id> - Отметить клиента как посещенного
/location <адрес> - Установить местоположение

<b>Настройки уведомлений:</b>
/notifications on/off - Включить/выключить
/remind <минуты> - Напоминания за N минут
/traffic <минуты> - Трафик от N минут
/delay <минуты> - Опоздания от N минут

<b>Примеры:</b>
/remind 15 - напоминания за 15 минут
/traffic 10 - уведомления о трафике от 10 минут
/visited 5 - отметить клиента 5 как посещенного
        """.strip()

        message = NotificationMessage(
            chat_id=chat_id,
            text=text,
            priority=3
        )

        return self.send_message(message)

class NotificationSystem:
    """Главная система уведомлений"""

    def __init__(self, bot_token: str):
        self.notifier = TelegramNotifier(bot_token)
        self.user_settings: Dict[int, UserSettings] = {}

    def register_user(self, chat_id: int, settings: UserSettings):
        """Регистрирует пользователя в системе уведомлений"""
        self.user_settings[chat_id] = settings
        print(f"👤 Пользователь {chat_id} зарегистрирован в системе уведомлений")

    def handle_trigger(self, trigger: TimeTrigger):
        """Обрабатывает триггер и отправляет уведомление"""
        chat_id = trigger.data.get("chat_id")
        if not chat_id or chat_id not in self.user_settings:
            return

        settings = self.user_settings[chat_id]
        if not settings.notifications_enabled:
            return

        success = False

        if trigger.trigger_type == TriggerType.DEPARTURE_REMINDER:
            success = self.notifier.send_departure_reminder(
                chat_id,
                trigger.data["client_info"],
                trigger.data["departure_time"]
            )
        elif trigger.trigger_type == TriggerType.LUNCH_BREAK:
            success = self.notifier.send_lunch_reminder(
                chat_id,
                trigger.data["lunch_start"],
                trigger.data["lunch_end"]
            )
        elif trigger.trigger_type == TriggerType.DELAY_ALERT:
            success = self.notifier.send_delay_alert(
                chat_id,
                trigger.data["client_info"],
                trigger.data["delay_minutes"],
                trigger.data["planned_arrival"]
            )
        elif trigger.trigger_type == TriggerType.TRAFFIC_CHANGE:
            success = self.notifier.send_traffic_alert(
                chat_id,
                trigger.data["route_info"],
                trigger.data["time_increase"],
                trigger.data["old_time"],
                trigger.data["new_time"]
            )
        elif trigger.trigger_type == TriggerType.ROUTE_UPDATE:
            success = self.notifier.send_route_update(
                chat_id,
                trigger.data["route_info"]
            )
        elif trigger.trigger_type == TriggerType.CLIENT_ARRIVAL:
            success = self.notifier.send_client_arrival(
                chat_id,
                trigger.data["client_info"],
                trigger.data["arrival_time"]
            )

        if success:
            print(f"✅ Уведомление {trigger.trigger_type.value} отправлено пользователю {chat_id}")
        else:
            print(f"❌ Ошибка отправки уведомления {trigger.trigger_type.value} пользователю {chat_id}")

    def send_manual_notification(self, chat_id: int, text: str) -> bool:
        """Отправляет ручное уведомление"""
        message = NotificationMessage(
            chat_id=chat_id,
            text=text,
            priority=3
        )
        return self.notifier.send_message(message)

    def get_user_settings(self, chat_id: int) -> Optional[UserSettings]:
        """Возвращает настройки пользователя"""
        return self.user_settings.get(chat_id)

    def update_user_settings(self, chat_id: int, **kwargs):
        """Обновляет настройки пользователя"""
        if chat_id in self.user_settings:
            for key, value in kwargs.items():
                if hasattr(self.user_settings[chat_id], key):
                    setattr(self.user_settings[chat_id], key, value)
            print(f"⚙️ Настройки пользователя {chat_id} обновлены")

def main():
    """Тестирование системы уведомлений"""
    print("🔔 Тестирование системы уведомлений")
    print("=" * 60)

    # Замените на ваш токен бота
    BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"

    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ Установите токен бота в переменной BOT_TOKEN")
        return

    notification_system = NotificationSystem(BOT_TOKEN)

    # Тестовый пользователь
    test_chat_id = 123456789
    test_settings = UserSettings(
        chat_id=test_chat_id,
        notifications_enabled=True,
        departure_reminder_minutes=15,
        lunch_reminder_minutes=15,
        delay_threshold_minutes=5,
        traffic_threshold_minutes=10
    )

    notification_system.register_user(test_chat_id, test_settings)

    # Тестируем отправку уведомлений
    print("📱 Тестирование отправки уведомлений...")

    # Тест 1: Напоминание о выезде
    client_info = {
        "id": 1,
        "address": "Ростов-на-Дону, ул. Большая Садовая, 1",
        "client_level": "VIP",
        "travel_time": 20
    }

    success = notification_system.notifier.send_departure_reminder(
        test_chat_id,
        client_info,
        datetime.now() + timedelta(minutes=15)
    )

    if success:
        print("✅ Тест напоминания о выезде прошел успешно")
    else:
        print("❌ Тест напоминания о выезде не прошел")

    # Тест 2: Меню настроек
    success = notification_system.notifier.send_settings_menu(test_chat_id, test_settings)

    if success:
        print("✅ Тест меню настроек прошел успешно")
    else:
        print("❌ Тест меню настроек не прошел")

    print("✅ Тестирование завершено")

if __name__ == "__main__":
    main()
