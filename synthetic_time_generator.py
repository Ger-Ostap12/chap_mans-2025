#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор синтетических временных данных для дообучения модели
"""

import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SyntheticTimeGenerator:
    """
    Генератор синтетических временных данных для клиентов
    """

    def __init__(self, seed: int = 42):
        """
        Инициализация генератора
        """
        random.seed(seed)
        np.random.seed(seed)
        print("🕒 Инициализация генератора временных данных...")

    def generate_time_windows(self, num_clients: int = 50000):
        """
        Генерирует временные окна для клиентов
        """
        print(f"⏰ Генерируем {num_clients} временных окон...")

        time_data = []

        for i in range(num_clients):
            # Генерируем случайные сдвиги
            work_start_shift = random.randint(0, 4)  # 0-4 часа
            work_end_shift = random.randint(0, 4)   # 0-4 часа
            lunch_start_shift = random.randint(0, 1) # 0-1 час
            lunch_end_shift = random.randint(0, 1)   # 0-1 час

            # Вычисляем времена
            work_start = self._add_hours("08:00", work_start_shift)
            work_end = self._subtract_hours("18:00", work_end_shift)
            lunch_start = self._add_hours("13:00", lunch_start_shift)
            lunch_end = self._add_hours("14:00", lunch_end_shift)

            # Проверяем логичность (work_start < work_end)
            if self._time_to_minutes(work_start) >= self._time_to_minutes(work_end):
                # Если некорректно, исправляем
                work_start = "08:00"
                work_end = "18:00"

            # Проверяем, что ланч внутри рабочего времени
            if (self._time_to_minutes(lunch_start) < self._time_to_minutes(work_start) or
                self._time_to_minutes(lunch_end) > self._time_to_minutes(work_end)):
                # Корректируем ланч
                lunch_start = "13:00"
                lunch_end = "14:00"

            time_data.append({
                'work_start_hour': self._time_to_minutes(work_start) / 60.0,  # Нормализованные часы
                'work_end_hour': self._time_to_minutes(work_end) / 60.0,
                'lunch_start_hour': self._time_to_minutes(lunch_start) / 60.0,
                'lunch_end_hour': self._time_to_minutes(lunch_end) / 60.0
            })

        print(f"✅ Сгенерировано {len(time_data)} временных окон")
        return time_data

    def _add_hours(self, time_str: str, hours: int) -> str:
        """
        Добавляет часы к времени
        """
        time_obj = datetime.strptime(time_str, "%H:%M")
        new_time = time_obj + timedelta(hours=hours)
        return new_time.strftime("%H:%M")

    def _subtract_hours(self, time_str: str, hours: int) -> str:
        """
        Вычитает часы из времени
        """
        time_obj = datetime.strptime(time_str, "%H:%M")
        new_time = time_obj - timedelta(hours=hours)
        return new_time.strftime("%H:%M")

    def _time_to_minutes(self, time_str: str) -> int:
        """
        Преобразует время в минуты
        """
        time_obj = datetime.strptime(time_str, "%H:%M")
        return time_obj.hour * 60 + time_obj.minute

    def combine_with_nyc_data(self, nyc_data_path: str = "DS/taxi_trip_data.csv", output_path: str = "combined_training_data.csv"):
        """
        Объединяет NYC данные с синтетическими временными данными
        """
        print("🔄 Объединяем NYC данные с синтетическими временными данными...")

        # Загружаем NYC данные
        if not os.path.exists(nyc_data_path):
            print(f"❌ Файл NYC данных не найден: {nyc_data_path}")
            return None

        print("📊 Загружаем NYC данные...")
        nyc_trips = []
        chunk_size = 10000

        try:
            for chunk in pd.read_csv(nyc_data_path, chunksize=chunk_size):
                if len(nyc_trips) >= 50000:  # Ограничиваем до 50k
                    break

                for _, row in chunk.iterrows():
                    try:
                        pickup_location = row.get('pickup_location_id', None)
                        dropoff_location = row.get('dropoff_location_id', None)
                        trip_distance = row.get('trip_distance', 0)
                        fare_amount = row.get('fare_amount', 0)

                        if (pd.notna(pickup_location) and pd.notna(dropoff_location) and
                            pickup_location > 0 and dropoff_location > 0):

                            trip = {
                                'latitude': pickup_location / 1000.0,
                                'longitude': dropoff_location / 1000.0,
                                'is_vip': random.choice([True, False]),  # Случайный VIP статус
                                'timestamp': np.random.randint(0, 86400),
                                'distance': float(trip_distance) if pd.notna(trip_distance) else 1.0,
                                'time': np.random.uniform(5, 120),
                                'fare': float(fare_amount) if pd.notna(fare_amount) else 10.0
                            }
                            nyc_trips.append(trip)

                    except Exception as e:
                        continue

        except Exception as e:
            print(f"❌ Ошибка загрузки NYC данных: {e}")
            return None

        print(f"✅ Загружено {len(nyc_trips)} NYC поездок")

        # Генерируем временные данные
        time_data = self.generate_time_windows(len(nyc_trips))

        # Объединяем данные
        print("🔗 Объединяем данные...")
        combined_data = []

        for i, trip in enumerate(nyc_trips):
            time_info = time_data[i]

            combined_record = {
                'latitude': trip['latitude'],
                'longitude': trip['longitude'],
                'is_vip': trip['is_vip'],
                'work_start_hour': time_info['work_start_hour'],
                'work_end_hour': time_info['work_end_hour'],
                'lunch_start_hour': time_info['lunch_start_hour'],
                'lunch_end_hour': time_info['lunch_end_hour']
            }

            combined_data.append(combined_record)

        # Сохраняем объединенные данные
        df = pd.DataFrame(combined_data)
        df.to_csv(output_path, index=False)

        print(f"✅ Объединенные данные сохранены: {output_path}")
        print(f"📊 Записей: {len(df)}")
        print(f"📋 Колонки: {list(df.columns)}")

        return output_path

def main():
    """
    Основная функция генерации данных
    """
    print("🕒 Генерация синтетических временных данных")
    print("=" * 60)

    # Создаем генератор
    generator = SyntheticTimeGenerator()

    # Объединяем данные
    output_file = generator.combine_with_nyc_data()

    if output_file:
        print(f"\n🎉 Генерация завершена!")
        print(f"📁 Файл: {output_file}")
        print(f"📊 Готов для дообучения модели")
    else:
        print(f"\n❌ Ошибка генерации данных")

if __name__ == "__main__":
    main()
