#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⏰ Система мониторинга времени для уведомлений
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class TriggerType(Enum):
    """Типы триггеров уведомлений"""
    DEPARTURE_REMINDER = "departure_reminder"  # Напоминание о выезде
    LUNCH_BREAK = "lunch_break"                # Обеденный перерыв
    DELAY_ALERT = "delay_alert"                # Опоздание
    TRAFFIC_CHANGE = "traffic_change"          # Изменение трафика
    ROUTE_UPDATE = "route_update"              # Обновление маршрута
    CLIENT_ARRIVAL = "client_arrival"          # Прибытие к клиенту

@dataclass
class TimeTrigger:
    """Триггер времени"""
    trigger_type: TriggerType
    trigger_time: datetime
    message: str
    data: Dict
    priority: int = 1  # 1-высокий, 2-средний, 3-низкий

@dataclass
class UserSettings:
    """Настройки пользователя для уведомлений"""
    chat_id: int
    notifications_enabled: bool = True
    departure_reminder_minutes: int = 15
    lunch_reminder_minutes: int = 15
    delay_threshold_minutes: int = 5
    traffic_threshold_minutes: int = 10
    work_start: str = "09:00"
    work_end: str = "18:00"
    lunch_start: str = "13:00"
    lunch_end: str = "14:00"
    timezone: str = "Europe/Moscow"
    language: str = "ru"

class TimeMonitor:
    """Монитор времени для системы уведомлений"""

    def __init__(self):
        self.triggers: List[TimeTrigger] = []
        self.user_settings: Dict[int, UserSettings] = {}
        self.callbacks: Dict[TriggerType, List[Callable]] = {
            TriggerType.DEPARTURE_REMINDER: [],
            TriggerType.LUNCH_BREAK: [],
            TriggerType.DELAY_ALERT: [],
            TriggerType.TRAFFIC_CHANGE: [],
            TriggerType.ROUTE_UPDATE: [],
            TriggerType.CLIENT_ARRIVAL: []
        }
        self.monitoring = False
        self.monitor_thread = None

    def add_user(self, chat_id: int, settings: UserSettings):
        """Добавляет пользователя в систему мониторинга"""
        self.user_settings[chat_id] = settings
        print(f"👤 Пользователь {chat_id} добавлен в систему мониторинга")

    def update_user_settings(self, chat_id: int, **kwargs):
        """Обновляет настройки пользователя"""
        if chat_id in self.user_settings:
            for key, value in kwargs.items():
                if hasattr(self.user_settings[chat_id], key):
                    setattr(self.user_settings[chat_id], key, value)
            print(f"⚙️ Настройки пользователя {chat_id} обновлены")

    def register_callback(self, trigger_type: TriggerType, callback: Callable):
        """Регистрирует callback для типа триггера"""
        self.callbacks[trigger_type].append(callback)
        print(f"🔔 Callback зарегистрирован для {trigger_type.value}")

    def add_trigger(self, trigger: TimeTrigger):
        """Добавляет триггер в очередь"""
        self.triggers.append(trigger)
        # Сортируем по времени срабатывания
        self.triggers.sort(key=lambda x: x.trigger_time)
        print(f"⏰ Триггер добавлен: {trigger.trigger_type.value} в {trigger.trigger_time.strftime('%H:%M')}")

    def add_departure_reminder(self, chat_id: int, departure_time: datetime, client_info: Dict):
        """Добавляет напоминание о выезде"""
        if chat_id not in self.user_settings:
            return

        settings = self.user_settings[chat_id]
        if not settings.notifications_enabled:
            return

        reminder_time = departure_time - timedelta(minutes=settings.departure_reminder_minutes)

        trigger = TimeTrigger(
            trigger_type=TriggerType.DEPARTURE_REMINDER,
            trigger_time=reminder_time,
            message=f"⏰ Через {settings.departure_reminder_minutes} мин выезжайте к клиенту",
            data={
                "chat_id": chat_id,
                "client_info": client_info,
                "departure_time": departure_time
            },
            priority=2
        )

        self.add_trigger(trigger)

    def add_lunch_reminder(self, chat_id: int, lunch_start: datetime):
        """Добавляет напоминание об обеде"""
        if chat_id not in self.user_settings:
            return

        settings = self.user_settings[chat_id]
        if not settings.notifications_enabled:
            return

        reminder_time = lunch_start - timedelta(minutes=settings.lunch_reminder_minutes)

        trigger = TimeTrigger(
            trigger_type=TriggerType.LUNCH_BREAK,
            trigger_time=reminder_time,
            message=f"🍽️ Время обеда через {settings.lunch_reminder_minutes} минут с {settings.lunch_start} до {settings.lunch_end}",
            data={
                "chat_id": chat_id,
                "lunch_start": lunch_start,
                "lunch_end": datetime.strptime(settings.lunch_end, "%H:%M").time()
            },
            priority=2
        )

        self.add_trigger(trigger)

    def add_delay_alert(self, chat_id: int, planned_arrival: datetime, current_time: datetime, client_info: Dict):
        """Добавляет уведомление об опоздании"""
        if chat_id not in self.user_settings:
            return

        settings = self.user_settings[chat_id]
        if not settings.notifications_enabled:
            return

        delay_minutes = int((current_time - planned_arrival).total_seconds() / 60)

        if delay_minutes >= settings.delay_threshold_minutes:
            trigger = TimeTrigger(
                trigger_type=TriggerType.DELAY_ALERT,
                trigger_time=current_time,
                message=f"🚨 Вы опаздываете на {delay_minutes} мин к клиенту",
                data={
                    "chat_id": chat_id,
                    "client_info": client_info,
                    "delay_minutes": delay_minutes,
                    "planned_arrival": planned_arrival
                },
                priority=1
            )

            self.add_trigger(trigger)

    def add_traffic_change_alert(self, chat_id: int, old_time: int, new_time: int, route_info: Dict):
        """Добавляет уведомление об изменении трафика"""
        if chat_id not in self.user_settings:
            return

        settings = self.user_settings[chat_id]
        if not settings.notifications_enabled:
            return

        time_increase = new_time - old_time

        if time_increase >= settings.traffic_threshold_minutes:
            trigger = TimeTrigger(
                trigger_type=TriggerType.TRAFFIC_CHANGE,
                trigger_time=datetime.now(),
                message=f"🚗 Пробка на маршруте! Время увеличено на {time_increase} мин",
                data={
                    "chat_id": chat_id,
                    "route_info": route_info,
                    "old_time": old_time,
                    "new_time": new_time,
                    "time_increase": time_increase
                },
                priority=1
            )

            self.add_trigger(trigger)

    def add_route_update_notification(self, chat_id: int, route_info: Dict):
        """Добавляет уведомление об обновлении маршрута"""
        if chat_id not in self.user_settings:
            return

        settings = self.user_settings[chat_id]
        if not settings.notifications_enabled:
            return

        trigger = TimeTrigger(
            trigger_type=TriggerType.ROUTE_UPDATE,
            trigger_time=datetime.now(),
            message="🔄 Маршрут обновлен с нового местоположения",
            data={
                "chat_id": chat_id,
                "route_info": route_info
            },
            priority=2
        )

        self.add_trigger(trigger)

    def add_client_arrival_notification(self, chat_id: int, client_info: Dict, arrival_time: datetime):
        """Добавляет уведомление о прибытии к клиенту"""
        if chat_id not in self.user_settings:
            return

        settings = self.user_settings[chat_id]
        if not settings.notifications_enabled:
            return

        trigger = TimeTrigger(
            trigger_type=TriggerType.CLIENT_ARRIVAL,
            trigger_time=arrival_time,
            message=f"📍 Вы прибыли к клиенту. Время обслуживания: {client_info.get('service_time', 20)} мин",
            data={
                "chat_id": chat_id,
                "client_info": client_info,
                "arrival_time": arrival_time
            },
            priority=2
        )

        self.add_trigger(trigger)

    def start_monitoring(self):
        """Запускает мониторинг времени"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("⏰ Мониторинг времени запущен")

    def stop_monitoring(self):
        """Останавливает мониторинг времени"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("⏰ Мониторинг времени остановлен")

    def _monitor_loop(self):
        """Основной цикл мониторинга"""
        while self.monitoring:
            try:
                current_time = datetime.now()

                # Проверяем триггеры, которые должны сработать
                triggers_to_fire = []
                for trigger in self.triggers[:]:
                    if trigger.trigger_time <= current_time:
                        triggers_to_fire.append(trigger)
                        self.triggers.remove(trigger)

                # Срабатываем триггеры
                for trigger in triggers_to_fire:
                    self._fire_trigger(trigger)

                # Спим 30 секунд перед следующей проверкой
                time.sleep(30)

            except Exception as e:
                print(f"❌ Ошибка в мониторинге времени: {e}")
                time.sleep(60)  # Ждем минуту при ошибке

    def _fire_trigger(self, trigger: TimeTrigger):
        """Срабатывает триггер"""
        print(f"🔔 Срабатывает триггер: {trigger.trigger_type.value}")
        print(f"📱 Сообщение: {trigger.message}")

        # Вызываем все зарегистрированные callbacks
        for callback in self.callbacks[trigger.trigger_type]:
            try:
                callback(trigger)
            except Exception as e:
                print(f"❌ Ошибка в callback: {e}")

    def get_user_triggers(self, chat_id: int) -> List[TimeTrigger]:
        """Возвращает триггеры пользователя"""
        return [trigger for trigger in self.triggers
                if trigger.data.get("chat_id") == chat_id]

    def clear_user_triggers(self, chat_id: int):
        """Очищает триггеры пользователя"""
        self.triggers = [trigger for trigger in self.triggers
                        if trigger.data.get("chat_id") != chat_id]
        print(f"🗑️ Триггеры пользователя {chat_id} очищены")

    def get_status(self) -> Dict:
        """Возвращает статус системы мониторинга"""
        return {
            "monitoring": self.monitoring,
            "total_triggers": len(self.triggers),
            "registered_users": len(self.user_settings),
            "next_trigger": self.triggers[0].trigger_time if self.triggers else None
        }

def main():
    """Тестирование системы мониторинга времени"""
    print("⏰ Тестирование системы мониторинга времени")
    print("=" * 60)

    monitor = TimeMonitor()

    # Добавляем тестового пользователя
    test_user = UserSettings(
        chat_id=123456789,
        notifications_enabled=True,
        departure_reminder_minutes=15,
        lunch_reminder_minutes=15,
        delay_threshold_minutes=5,
        traffic_threshold_minutes=10
    )
    monitor.add_user(test_user.chat_id, test_user)

    # Регистрируем callback для тестирования
    def test_callback(trigger: TimeTrigger):
        print(f"🎯 Callback сработал: {trigger.message}")

    monitor.register_callback(TriggerType.DEPARTURE_REMINDER, test_callback)
    monitor.register_callback(TriggerType.LUNCH_BREAK, test_callback)
    monitor.register_callback(TriggerType.DELAY_ALERT, test_callback)

    # Добавляем тестовые триггеры
    now = datetime.now()

    # Напоминание о выезде через 1 минуту
    departure_time = now + timedelta(minutes=1)
    monitor.add_departure_reminder(
        test_user.chat_id,
        departure_time,
        {"id": 1, "address": "Тестовый адрес"}
    )

    # Напоминание об обеде через 2 минуты
    lunch_time = now + timedelta(minutes=2)
    monitor.add_lunch_reminder(test_user.chat_id, lunch_time)

    # Запускаем мониторинг
    monitor.start_monitoring()

    print("⏰ Мониторинг запущен, ждем срабатывания триггеров...")
    print("⏰ Нажмите Ctrl+C для остановки")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏰ Остановка мониторинга...")
        monitor.stop_monitoring()
        print("✅ Тестирование завершено")

if __name__ == "__main__":
    main()
