#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестирование обученной модели на реальных данных клиентов
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from train_model import AttentionRouteOptimizer, RouteOptimizationDataset

class RouteBuilder:
    """
    Построитель маршрутов с использованием обученной модели
    """

    def __init__(self, model_path: str = "best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttentionRouteOptimizer().to(self.device)

        # Загружаем обученную модель
        if os.path.exists(model_path):
            print(f"📁 Загружаем модель из {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            print("✅ Модель загружена успешно")
        else:
            print(f"❌ Файл модели не найден: {model_path}")
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

    def load_client_data(self, data_file: str):
        """
        Загружает данные клиентов из файла
        """
        print(f"📊 Загружаем данные клиентов из {data_file}...")

        if not os.path.exists(data_file):
            print(f"❌ Файл не найден: {data_file}")
            return None

        try:
            # Читаем файл как текст
            with open(data_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Ищем JSON-подобную структуру
            if 'data = [' in content:
                print("🔍 Обнаружен JSON-подобный формат, парсим...")
                return self._parse_json_like_data(content)
            else:
                # Пробуем как CSV
                if data_file.endswith('.csv'):
                    df = pd.read_csv(data_file)
                elif data_file.endswith('.txt'):
                    # Пробуем как CSV с разделителем
                    df = pd.read_csv(data_file, sep='\t')
                    if len(df.columns) == 1:
                        df = pd.read_csv(data_file, sep=',')
                else:
                    print("❌ Неподдерживаемый формат файла")
                    return None

                print(f"📋 Колонки в данных: {list(df.columns)}")
                print(f"📊 Количество клиентов: {len(df)}")

                return df

        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            return None

    def _parse_json_like_data(self, content: str):
        """
        Парсит JSON-подобные данные из файла
        """
        import re
        import json

        try:
            # Ищем массив данных
            data_match = re.search(r'data = \[(.*?)\]', content, re.DOTALL)
            if not data_match:
                print("❌ Не найден массив данных")
                return None

            data_str = data_match.group(1)

            # Разбиваем на отдельные объекты
            objects = []
            current_obj = ""
            brace_count = 0

            for char in data_str:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1

                current_obj += char

                if brace_count == 0 and current_obj.strip():
                    # Завершили объект
                    obj_str = current_obj.strip()
                    if obj_str.startswith('{') and obj_str.endswith('}'):
                        objects.append(obj_str)
                    current_obj = ""

            print(f"📊 Найдено {len(objects)} клиентов")

            # Парсим каждый объект
            clients_data = []
            for i, obj_str in enumerate(objects):
                try:
                    # Очищаем от лишних символов
                    obj_str = obj_str.strip()
                    if obj_str.endswith(','):
                        obj_str = obj_str[:-1]

                    # Парсим JSON
                    client_data = json.loads(obj_str)
                    clients_data.append(client_data)

                except Exception as e:
                    print(f"⚠️  Ошибка парсинга клиента {i+1}: {e}")
                    continue

            # Преобразуем в DataFrame
            df = pd.DataFrame(clients_data)
            print(f"✅ Успешно загружено {len(df)} клиентов")
            print(f"📋 Колонки: {list(df.columns)}")

            return df

        except Exception as e:
            print(f"❌ Ошибка парсинга JSON-подобных данных: {e}")
            return None

    def prepare_client_features(self, df):
        """
        Подготавливает признаки клиентов для модели
        """
        print("🔧 Подготавливаем признаки клиентов...")

        clients = []

        for _, row in df.iterrows():
            try:
                # Извлекаем координаты (пробуем разные варианты названий)
                lat = float(row.get('lat', row.get('latitude', 0)))
                lon = float(row.get('lon', row.get('longitude', row.get('lng', 0))))

                # Извлекаем дополнительные параметры
                client_level = str(row.get('client_level', 'Стандарт'))
                is_vip = client_level == 'VIP'
                priority = 2 if is_vip else 1  # VIP = 2, обычный = 1

                # Имя клиента
                name = str(row.get('address1', row.get('address', f"Клиент {row.get('id', 'Unknown')}")))

                # Время работы
                work_start = str(row.get('work_start', '09:00'))
                work_end = str(row.get('work_end', '18:00'))

                # Создаем признаки клиента (7 параметров как ожидает модель)
                # [lat, lon, is_vip, work_start_hour, work_end_hour, lunch_start_hour, lunch_end_hour]
                work_start_hour = float(work_start.split(':')[0]) / 24.0  # Нормализуем время
                work_end_hour = float(work_end.split(':')[0]) / 24.0
                lunch_start_hour = float(str(row.get('lunch_start', '13:00')).split(':')[0]) / 24.0
                lunch_end_hour = float(str(row.get('lunch_end', '14:00')).split(':')[0]) / 24.0

                client = {
                    'id': int(row.get('id', 0)),
                    'name': name,
                    'address': str(row.get('address', '')),
                    'latitude': lat,
                    'longitude': lon,
                    'is_vip': is_vip,
                    'priority': priority,
                    'client_level': client_level,
                    'work_start': work_start,
                    'work_end': work_end,
                    'features': [lat, lon, float(is_vip), work_start_hour, work_end_hour, lunch_start_hour, lunch_end_hour]
                }

                clients.append(client)

            except Exception as e:
                print(f"⚠️  Ошибка обработки клиента: {e}")
                continue

        print(f"✅ Подготовлено {len(clients)} клиентов")
        return clients

    def build_route(self, clients):
        """
        Строит оптимальный маршрут для клиентов
        """
        if not clients:
            print("❌ Нет клиентов для построения маршрута")
            return None

        print(f"🗺️  Строим маршрут для {len(clients)} клиентов...")

        # Подготавливаем данные для модели
        features = []
        for client in clients:
            features.append(client['features'])

        # Создаем тензор
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Создаем маску
        batch_size, seq_len = features_tensor.shape[:2]
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)

        # Получаем предсказания модели
        with torch.no_grad():
            outputs = self.model(features_tensor, mask)
            route_scores = outputs['route_scores'].cpu().numpy()[0]

        # Сортируем клиентов по предсказанным очкам
        client_scores = list(zip(clients, route_scores))
        client_scores.sort(key=lambda x: x[1], reverse=True)

        # Создаем оптимальный маршрут
        optimal_route = [client for client, score in client_scores]

        print("✅ Маршрут построен!")
        return optimal_route

    def print_route(self, route):
        """
        Выводит построенный маршрут
        """
        if not route:
            print("❌ Маршрут пуст")
            return

        print("\n🗺️  ОПТИМАЛЬНЫЙ МАРШРУТ:")
        print("=" * 80)

        total_distance = 0
        for i, client in enumerate(route, 1):
            vip_status = "🌟 VIP" if client['is_vip'] else "👤 Стандарт"
            priority = f"Приоритет: {client['priority']}"

            print(f"{i:2d}. ID: {client['id']} - {client['name']}")
            print(f"    📍 Координаты: {client['latitude']:.6f}, {client['longitude']:.6f}")
            print(f"    🏢 Адрес: {client['address']}")
            print(f"    {vip_status} | {priority}")
            print(f"    ⏰ Время работы: {client['work_start']} - {client['work_end']}")

            # Вычисляем расстояние до следующего клиента
            if i < len(route):
                next_client = route[i]
                distance = self._calculate_distance(
                    client['latitude'], client['longitude'],
                    next_client['latitude'], next_client['longitude']
                )
                total_distance += distance
                print(f"    📏 До следующего: {distance:.2f} км")

            print("-" * 80)

        print(f"\n📊 СТАТИСТИКА МАРШРУТА:")
        print(f"   🛣️  Общая длина: {total_distance:.2f} км")
        print(f"   👥 Всего клиентов: {len(route)}")

        # Статистика по VIP
        vip_count = sum(1 for client in route if client['is_vip'])
        standard_count = len(route) - vip_count
        print(f"   🌟 VIP клиентов: {vip_count}")
        print(f"   👤 Стандартных: {standard_count}")

        # Показываем первые 3 и последние 3 клиента
        if len(route) > 6:
            print(f"\n🎯 КЛЮЧЕВЫЕ ТОЧКИ:")
            print(f"   🚀 Начало: {route[0]['name']} ({'VIP' if route[0]['is_vip'] else 'Стандарт'})")
            print(f"   🏁 Финиш: {route[-1]['name']} ({'VIP' if route[-1]['is_vip'] else 'Стандарт'})")

        print(f"\n✅ Модель успешно построила оптимальный маршрут!")

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Вычисляет расстояние между двумя точками (упрощенная формула)
        """
        # Упрощенная формула расстояния
        return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111  # Примерно км

def main():
    """
    Основная функция тестирования
    """
    print("🧪 Тестирование обученной модели на реальных данных")
    print("=" * 60)

    # Проверяем наличие модели
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        print("💡 Сначала обучите модель: python train_model.py")
        return

    # Проверяем наличие данных клиентов
    data_file = "DATA (2).txt"
    if not os.path.exists(data_file):
        print(f"❌ Файл данных клиентов не найден: {data_file}")
        print("💡 Убедитесь, что файл с данными клиентов находится в корне проекта")
        return

    try:
        # Создаем построитель маршрутов
        route_builder = RouteBuilder(model_path)

        # Загружаем данные клиентов
        df = route_builder.load_client_data(data_file)
        if df is None:
            return

        # Подготавливаем признаки
        clients = route_builder.prepare_client_features(df)
        if not clients:
            print("❌ Не удалось подготовить данные клиентов")
            return

        # Строим маршрут
        route = route_builder.build_route(clients)
        if route is None:
            return

        # Выводим результат
        route_builder.print_route(route)

        print("\n🎉 Тестирование завершено успешно!")
        print("💡 Модель готова к использованию в вашем приложении!")

    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
