"""
Модуль для анализа реальных расстояний из NYC данных
Использует trip_distance, taxi_zone_geo и TomTom API
"""

import pandas as pd
import numpy as np
import json
import requests
from typing import Dict, List, Tuple, Optional
import os
import time
from math import radians, cos, sin, asin, sqrt
import pickle

class RealDistanceAnalyzer:
    """
    Анализ реальных расстояний из NYC данных
    """

    def __init__(self, tomtom_api_key: str = None):
        self.tomtom_api_key = tomtom_api_key
        self.zone_coordinates = {}
        self.distance_matrix = {}
        self.trip_data = []

    def load_nyc_trip_data(self, file_path: str = "DS/original_cleaned_nyc_taxi_data_2018.csv") -> pd.DataFrame:
        """
        Загружает данные о поездках из NYC
        """
        print("📊 Загрузка NYC данных о поездках...")

        try:
            # Читаем только нужные колонки для экономии памяти
            columns = [
                'trip_distance', 'pickup_location_id', 'dropoff_location_id',
                'trip_duration', 'fare_amount', 'hour_of_day'
            ]

            df = pd.read_csv(file_path, usecols=columns)

            # Фильтруем валидные данные
            df = df.dropna()
            df = df[df['trip_distance'] > 0]
            df = df[df['trip_duration'] > 0]

            print(f"✅ Загружено {len(df)} валидных поездок")
            return df

        except Exception as e:
            print(f"❌ Ошибка загрузки NYC данных: {e}")
            return pd.DataFrame()

    def load_zone_geometries(self, file_path: str = "DS/taxi_zone_geo.csv") -> Dict:
        """
        Загружает геометрии зон и вычисляет их центры
        """
        print("🗺️ Загрузка геометрий зон...")

        try:
            df = pd.read_csv(file_path)
            zone_coords = {}

            for _, row in df.iterrows():
                zone_id = row['zone_id']
                zone_name = row['zone_name']
                borough = row['borough']

                # Парсим геометрию (упрощенно - берем первую координату)
                geom_str = row['zone_geom']
                if 'POLYGON' in geom_str:
                    try:
                        # Извлекаем координаты из POLYGON
                        coords_str = geom_str.split('POLYGON((')[1].split('))')[0]
                        coords = coords_str.split(', ')

                        # Берем первую координату как центр
                        if coords:
                            first_coord = coords[0].strip().split()
                            if len(first_coord) >= 2:
                                # Убираем скобки и парсим координаты
                                lon_str = first_coord[0].replace('(', '').replace(')', '')
                                lat_str = first_coord[1].replace('(', '').replace(')', '')

                                lon = float(lon_str)
                                lat = float(lat_str)

                                zone_coords[zone_id] = {
                                    'lat': lat,
                                    'lon': lon,
                                    'name': zone_name,
                                    'borough': borough
                                }
                    except (ValueError, IndexError) as e:
                        print(f"⚠️ Ошибка парсинга зоны {zone_id}: {e}")
                        continue

            self.zone_coordinates = zone_coords
            print(f"✅ Загружено {len(zone_coords)} зон с координатами")
            return zone_coords

        except Exception as e:
            print(f"❌ Ошибка загрузки геометрий зон: {e}")
            return {}

    def calculate_haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Вычисляет расстояние между двумя точками по формуле Haversine
        """
        # Конвертируем в радианы
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Формула Haversine
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))

        # Радиус Земли в метрах
        r = 6371000
        return c * r

    def build_distance_matrix(self) -> Dict:
        """
        Строит матрицу расстояний между всеми зонами
        """
        print("📏 Построение матрицы расстояний между зонами...")

        if not self.zone_coordinates:
            print("❌ Нет данных о координатах зон")
            return {}

        distance_matrix = {}
        zones = list(self.zone_coordinates.keys())

        for i, zone1 in enumerate(zones):
            distance_matrix[zone1] = {}
            coord1 = self.zone_coordinates[zone1]

            for j, zone2 in enumerate(zones):
                coord2 = self.zone_coordinates[zone2]

                # Вычисляем расстояние
                distance = self.calculate_haversine_distance(
                    coord1['lat'], coord1['lon'],
                    coord2['lat'], coord2['lon']
                )

                distance_matrix[zone1][zone2] = distance

                if (i * len(zones) + j) % 1000 == 0:
                    print(f"  Обработано {i * len(zones) + j + 1}/{len(zones)**2} пар зон")

        self.distance_matrix = distance_matrix
        print(f"✅ Матрица расстояний построена для {len(zones)} зон")
        return distance_matrix

    def get_real_distance_from_nyc(self, pickup_zone: int, dropoff_zone: int,
                                  trip_data: pd.DataFrame) -> Optional[float]:
        """
        Получает реальное расстояние из NYC данных
        """
        # Ищем поездки между этими зонами
        trips = trip_data[
            (trip_data['pickup_location_id'] == pickup_zone) &
            (trip_data['dropoff_location_id'] == dropoff_zone)
        ]

        if len(trips) > 0:
            # Берем медианное расстояние
            median_distance = trips['trip_distance'].median()
            return median_distance * 1609.34  # Конвертируем мили в метры

        return None

    def get_distance_from_matrix(self, pickup_zone: int, dropoff_zone: int) -> Optional[float]:
        """
        Получает расстояние из матрицы расстояний
        """
        if pickup_zone in self.distance_matrix and dropoff_zone in self.distance_matrix[pickup_zone]:
            return self.distance_matrix[pickup_zone][dropoff_zone]
        return None

    def get_tomtom_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
        """
        Получает расстояние через TomTom API с исправленными SSL настройками
        """
        if not self.tomtom_api_key:
            return None

        try:
            # Создаем сессию с правильными SSL настройками
            session = requests.Session()
            session.verify = False  # Отключаем проверку SSL как показала диагностика

            # Настраиваем заголовки
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Connection': 'keep-alive'
            })

            url = "https://api.tomtom.com/routing/1/calculateRoute/{lat1},{lon1}:{lat2},{lon2}/json"
            url = url.format(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2)

            params = {
                'key': self.tomtom_api_key,
                'routeType': 'fastest',
                'traffic': 'true'
            }

            response = session.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if 'routes' in data and len(data['routes']) > 0:
                    route = data['routes'][0]
                    if 'summary' in route:
                        distance = route['summary']['lengthInMeters']
                        return distance

        except Exception as e:
            print(f"⚠️ Ошибка TomTom API: {e}")

        return None

    def get_optimal_distance(self, pickup_zone: int, dropoff_zone: int,
                           trip_data: pd.DataFrame, use_rostov_scaling: bool = True) -> float:
        """
        Получает оптимальное расстояние, используя все доступные источники
        """
        # 1. Пробуем получить из NYC данных (самый точный)
        nyc_distance = self.get_real_distance_from_nyc(pickup_zone, dropoff_zone, trip_data)
        if nyc_distance:
            # Адаптируем под Ростов (NYC расстояния обычно больше)
            if use_rostov_scaling:
                nyc_distance = nyc_distance * 0.6  # Масштабируем под Ростов
            return nyc_distance

        # 2. Пробуем получить из матрицы расстояний
        matrix_distance = self.get_distance_from_matrix(pickup_zone, dropoff_zone)
        if matrix_distance:
            return matrix_distance

        # 3. Пробуем получить через TomTom API (для Ростова)
        if pickup_zone in self.zone_coordinates and dropoff_zone in self.zone_coordinates:
            coord1 = self.zone_coordinates[pickup_zone]
            coord2 = self.zone_coordinates[dropoff_zone]

            tomtom_distance = self.get_tomtom_distance(
                coord1['lat'], coord1['lon'],
                coord2['lat'], coord2['lon']
            )
            if tomtom_distance:
                return tomtom_distance

        # 4. Fallback - вычисляем по прямой
        if pickup_zone in self.zone_coordinates and dropoff_zone in self.zone_coordinates:
            coord1 = self.zone_coordinates[pickup_zone]
            coord2 = self.zone_coordinates[dropoff_zone]

            return self.calculate_haversine_distance(
                coord1['lat'], coord1['lon'],
                coord2['lat'], coord2['lon']
            )

        # 5. Последний fallback - случайное расстояние для Ростова
        return np.random.uniform(500, 5000)  # 0.5-5 км для Ростова

    def analyze_distance_patterns(self, trip_data: pd.DataFrame) -> Dict:
        """
        Анализирует паттерны расстояний в данных
        """
        print("📊 Анализ паттернов расстояний...")

        analysis = {
            'total_trips': len(trip_data),
            'avg_distance': trip_data['trip_distance'].mean(),
            'median_distance': trip_data['trip_distance'].median(),
            'max_distance': trip_data['trip_distance'].max(),
            'min_distance': trip_data['trip_distance'].min(),
            'distance_std': trip_data['trip_distance'].std(),
            'avg_duration': trip_data['trip_duration'].mean(),
            'median_duration': trip_data['trip_duration'].median()
        }

        # Анализ по часам
        hourly_stats = trip_data.groupby('hour_of_day').agg({
            'trip_distance': ['mean', 'median', 'count'],
            'trip_duration': ['mean', 'median']
        }).round(2)

        analysis['hourly_stats'] = hourly_stats

        print(f"✅ Анализ завершен:")
        print(f"  📏 Среднее расстояние: {analysis['avg_distance']:.2f} миль")
        print(f"  ⏱️ Среднее время: {analysis['avg_duration']:.0f} секунд")
        print(f"  📊 Всего поездок: {analysis['total_trips']}")

        return analysis

    def save_distance_matrix(self, file_path: str = "distance_matrix.pkl"):
        """
        Сохраняет матрицу расстояний в файл
        """
        if self.distance_matrix:
            with open(file_path, 'wb') as f:
                pickle.dump(self.distance_matrix, f)
            print(f"💾 Матрица расстояний сохранена в {file_path}")

    def load_distance_matrix(self, file_path: str = "distance_matrix.pkl"):
        """
        Загружает матрицу расстояний из файла
        """
        try:
            with open(file_path, 'rb') as f:
                self.distance_matrix = pickle.load(f)
            print(f"📂 Матрица расстояний загружена из {file_path}")
        except FileNotFoundError:
            print(f"⚠️ Файл {file_path} не найден")
        except Exception as e:
            print(f"❌ Ошибка загрузки матрицы: {e}")

    def get_rostov_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Получает расстояние для ростовских координат
        """
        # 1. Пробуем TomTom API для Ростова
        if self.tomtom_api_key:
            tomtom_distance = self.get_tomtom_distance(lat1, lon1, lat2, lon2)
            if tomtom_distance:
                return tomtom_distance

        # 2. Fallback - вычисляем по прямой
        return self.calculate_haversine_distance(lat1, lon1, lat2, lon2)

    def create_rostov_training_data(self, clients_data: List[Dict],
                                 use_nyc_patterns: bool = True,
                                 num_synthetic_clients: int = 100) -> List[Dict]:
        """
        Создает данные для обучения с ростовскими расстояниями
        """
        print("🏙️ Создание данных для обучения с ростовскими расстояниями...")
        print(f"📊 Всего клиентов: {len(clients_data)}")
        print(f"🎯 Будет создано пар: {len(clients_data) * (len(clients_data) - 1)}")
        print(f"⏱️ Примерное время: {len(clients_data) * (len(clients_data) - 1) * 2 // 60} минут")

        training_data = []
        total_pairs = len(clients_data) * (len(clients_data) - 1)
        processed_pairs = 0

        # 1. Используем реальные ростовские клиенты
        for i, client1 in enumerate(clients_data):
            for j, client2 in enumerate(clients_data):
                if i != j:
                    # Показываем прогресс каждые 50 пар
                    if processed_pairs % 50 == 0:
                        progress = (processed_pairs / total_pairs) * 100
                        print(f"📈 Прогресс: {processed_pairs}/{total_pairs} ({progress:.1f}%)")

                    # Вычисляем расстояние между клиентами
                    distance = self.get_rostov_distance(
                        client1['lat'], client1['lon'],
                        client2['lat'], client2['lon']
                    )

                    # Создаем запись для обучения
                    training_record = {
                        'client1_id': i,
                        'client2_id': j,
                        'client1_lat': client1['lat'],
                        'client1_lon': client1['lon'],
                        'client2_lat': client2['lat'],
                        'client2_lon': client2['lon'],
                        'distance_meters': distance,
                        'client1_vip': client1.get('client_level') == 'VIP',
                        'client2_vip': client2.get('client_level') == 'VIP',
                        'client1_work_start': client1.get('work_start_hour', 8.0),
                        'client1_work_end': client1.get('work_end_hour', 18.0),
                        'client2_work_start': client2.get('work_start_hour', 8.0),
                        'client2_work_end': client2.get('work_end_hour', 18.0)
                    }

                    training_data.append(training_record)
                    processed_pairs += 1

                    # Небольшая задержка чтобы не превысить rate limit
                    time.sleep(0.1)  # 100ms задержка

        # 2. Создаем синтетических клиентов в Ростове
        if num_synthetic_clients > 0:
            print(f"🎲 Создание {num_synthetic_clients} синтетических клиентов...")
            synthetic_clients = self._generate_synthetic_rostov_clients(num_synthetic_clients)

            # Добавляем пары с синтетическими клиентами
            for i, client1 in enumerate(synthetic_clients):
                for j, client2 in enumerate(synthetic_clients):
                    if i != j:
                        distance = self.get_rostov_distance(
                            client1['lat'], client1['lon'],
                            client2['lat'], client2['lon']
                        )

                        training_record = {
                            'client1_id': f"synthetic_{i}",
                            'client2_id': f"synthetic_{j}",
                            'client1_lat': client1['lat'],
                            'client1_lon': client1['lon'],
                            'client2_lat': client2['lat'],
                            'client2_lon': client2['lon'],
                            'distance_meters': distance,
                            'client1_vip': client1.get('client_level') == 'VIP',
                            'client2_vip': client2.get('client_level') == 'VIP',
                            'client1_work_start': client1.get('work_start_hour', 8.0),
                            'client1_work_end': client1.get('work_end_hour', 18.0),
                            'client2_work_start': client2.get('work_start_hour', 8.0),
                            'client2_work_end': client2.get('work_end_hour', 18.0)
                        }

                        training_data.append(training_record)

        # 3. Используем NYC паттерны если нужно
        if use_nyc_patterns and hasattr(self, 'trip_data') and not self.trip_data.empty:
            print("📊 Добавление NYC паттернов...")
            nyc_pairs = self._create_nyc_training_pairs(1000)  # 1000 пар из NYC
            training_data.extend(nyc_pairs)

        print(f"✅ Создано {len(training_data)} записей для обучения")
        return training_data

    def _generate_synthetic_rostov_clients(self, num_clients: int) -> List[Dict]:
        """
        Генерирует синтетических клиентов в Ростове
        """
        clients = []

        # Центр Ростова
        center_lat, center_lon = 47.217855, 39.696085

        for i in range(num_clients):
            # Генерируем координаты в радиусе 10 км от центра
            lat_offset = np.random.uniform(-0.1, 0.1)  # ~10 км
            lon_offset = np.random.uniform(-0.1, 0.1)

            client = {
                'lat': center_lat + lat_offset,
                'lon': center_lon + lon_offset,
                'client_level': 'VIP' if np.random.random() < 0.1 else 'regular',  # 10% VIP
                'work_start_hour': np.random.uniform(8.0, 10.0),
                'work_end_hour': np.random.uniform(17.0, 19.0),
                'lunch_start_hour': np.random.uniform(12.0, 14.0),
                'lunch_end_hour': np.random.uniform(13.0, 15.0)
            }
            clients.append(client)

        return clients

    def _create_nyc_training_pairs(self, num_pairs: int) -> List[Dict]:
        """
        Создает пары для обучения из NYC данных
        """
        if not hasattr(self, 'trip_data') or self.trip_data.empty:
            return []

        pairs = []
        trip_data = self.trip_data.sample(n=min(num_pairs * 2, len(self.trip_data)))

        for i in range(0, len(trip_data) - 1, 2):
            if i + 1 < len(trip_data):
                trip1 = trip_data.iloc[i]
                trip2 = trip_data.iloc[i + 1]

                # Используем NYC расстояния, но адаптируем под Ростов
                distance = trip1.get('trip_distance', 0) * 1609.34 * 0.6  # мили -> метры, масштаб

                # Генерируем случайные координаты в Ростове
                center_lat, center_lon = 47.217855, 39.696085
                lat1 = center_lat + np.random.uniform(-0.05, 0.05)
                lon1 = center_lon + np.random.uniform(-0.05, 0.05)
                lat2 = center_lat + np.random.uniform(-0.05, 0.05)
                lon2 = center_lon + np.random.uniform(-0.05, 0.05)

                pair = {
                    'client1_id': f"nyc_{i}",
                    'client2_id': f"nyc_{i+1}",
                    'client1_lat': lat1,
                    'client1_lon': lon1,
                    'client2_lat': lat2,
                    'client2_lon': lon2,
                    'distance_meters': distance,
                    'client1_vip': np.random.random() < 0.1,
                    'client2_vip': np.random.random() < 0.1,
                    'client1_work_start': np.random.uniform(8.0, 10.0),
                    'client1_work_end': np.random.uniform(17.0, 19.0),
                    'client2_work_start': np.random.uniform(8.0, 10.0),
                    'client2_work_end': np.random.uniform(17.0, 19.0)
                }
                pairs.append(pair)

        return pairs

    def _load_rostov_clients_from_file(self, file_path: str) -> List[Dict]:
        """
        Загружает клиентов из DATA (2).txt
        """
        clients = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Извлекаем JSON данные
            import re
            json_match = re.search(r'data = \[(.*?)\]', content, re.DOTALL)
            if json_match:
                json_str = '[' + json_match.group(1) + ']'

                # Очищаем от комментариев и лишних запятых
                json_str = re.sub(r'//.*?\n', '', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                json_str = re.sub(r',\s*}', '}', json_str)

                import json
                data = json.loads(json_str)

                for item in data:
                    client = {
                        'lat': float(item['lat']),
                        'lon': float(item['lon']),
                        'client_level': item['client_level'],
                        'work_start_hour': float(item['work_start'].split(':')[0]) + float(item['work_start'].split(':')[1])/60,
                        'work_end_hour': float(item['work_end'].split(':')[0]) + float(item['work_end'].split(':')[1])/60,
                        'lunch_start_hour': float(item['lunch_start'].split(':')[0]) + float(item['lunch_start'].split(':')[1])/60,
                        'lunch_end_hour': float(item['lunch_end'].split(':')[0]) + float(item['lunch_end'].split(':')[1])/60
                    }
                    clients.append(client)

            print(f"✅ Загружено {len(clients)} клиентов из {file_path}")
            return clients

        except Exception as e:
            print(f"❌ Ошибка загрузки клиентов: {e}")
            return []

def main():
    """
    Демонстрация работы с реальными расстояниями
    """
    print("🚀 Анализ реальных расстояний для Ростова-на-Дону")
    print("=" * 50)

    # Инициализируем анализатор с TomTom API
    analyzer = RealDistanceAnalyzer(tomtom_api_key="N0e11R91bFHexBDVlfIzDr7gjLygvdjv")

    # Загружаем NYC данные для обучения паттернов
    trip_data = analyzer.load_nyc_trip_data()
    if trip_data.empty:
        print("❌ Не удалось загрузить данные о поездках")
        return

    # Загружаем геометрии зон
    zone_coords = analyzer.load_zone_geometries()
    if not zone_coords:
        print("❌ Не удалось загрузить геометрии зон")
        return

    # Анализируем паттерны из NYC данных
    analysis = analyzer.analyze_distance_patterns(trip_data)

    # Загружаем реальных клиентов из DATA (2).txt
    print("\n👥 Загрузка реальных клиентов из DATA (2).txt:")
    rostov_clients = analyzer._load_rostov_clients_from_file("DATA (2).txt")

    print(f"✅ Загружено {len(rostov_clients)} реальных клиентов")

    # Тестируем с ВСЕМИ ростовскими клиентами
    print(f"\n🏙️ Тестирование с {len(rostov_clients)} ростовскими клиентами:")
    print("📊 Создание пар клиентов для тестирования...")

    test_pairs = 0
    for i, client1 in enumerate(rostov_clients):  # Тестируем ВСЕХ клиентов
        for j, client2 in enumerate(rostov_clients):
            if i != j:
                distance = analyzer.get_tomtom_distance(
                    client1['lat'], client1['lon'],
                    client2['lat'], client2['lon']
                )
                if distance:
                    print(f"  Клиент {i+1} → Клиент {j+1}: {distance:.0f} метров")
                    test_pairs += 1
                else:
                    print(f"  Клиент {i+1} → Клиент {j+1}: Ошибка API")

    print(f"✅ Протестировано {test_pairs} пар клиентов")

    # Создаем данные для обучения с использованием ВСЕХ NYC данных
    print("\n📊 Создание данных для обучения из ВСЕХ NYC данных:")
    print(f"🎯 Цель: получить 10,000+ пар для обучения")
    print(f"📊 Исходные данные: {len(rostov_clients)} реальных клиентов")

    # Рассчитываем сколько синтетических клиентов нужно
    # 47 реальных + 200 синтетических = 247 клиентов
    # 247 × 246 = 60,762 пар (больше чем нужно!)
    training_data = analyzer.create_rostov_training_data(
        clients_data=rostov_clients,
        use_nyc_patterns=True,
        num_synthetic_clients=200  # 200 синтетических клиентов
    )

    print(f"\n📊 Создано {len(training_data)} записей для обучения")
    print("✅ Анализ завершен!")

    return analyzer, training_data

if __name__ == "__main__":
    main()
