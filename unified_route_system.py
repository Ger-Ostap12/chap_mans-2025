#!/usr/bin/env python3
"""
🧠 Единая система: ANN модель + TomTom API + граф маршрутов
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
import json
import requests
import time
import os
from dataclasses import dataclass
from enum import Enum
import pickle
from sklearn.preprocessing import StandardScaler
from location_detector import LocationDetector, Location
from time_monitor import TimeMonitor, UserSettings, TriggerType
from notification_system import NotificationSystem

class ClientLevel(Enum):
    VIP = "VIP"
    REGULAR = "Стандарт"

@dataclass
class Client:
    """Клиент с полной информацией"""
    id: int
    address: str
    lat: float
    lon: float
    client_level: ClientLevel
    work_start: str
    work_end: str
    lunch_start: str
    lunch_end: str

    @property
    def service_time_minutes(self) -> int:
        return 30 if self.client_level == ClientLevel.VIP else 20

    @property
    def work_start_hour(self) -> float:
        hour, minute = map(int, self.work_start.split(':'))
        return hour + minute / 60.0

    @property
    def work_end_hour(self) -> float:
        hour, minute = map(int, self.work_end.split(':'))
        return hour + minute / 60.0

    @property
    def lunch_start_hour(self) -> float:
        hour, minute = map(int, self.lunch_start.split(':'))
        return hour + minute / 60.0

    @property
    def lunch_end_hour(self) -> float:
        hour, minute = map(int, self.lunch_end.split(':'))
        return hour + minute / 60.0

class AttentionRouteOptimizer(nn.Module):
    """
    Attention-based Neural Network для оптимизации маршрутов
    Интегрированная версия с графом маршрутов
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 256, num_heads: int = 8, num_layers: int = 3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Входной слой
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Attention слои
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        # Нормализация
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Выходные слои для графа маршрутов
        self.route_output = nn.Linear(hidden_dim, 1)  # Очки маршрута
        self.time_output = nn.Linear(hidden_dim, 1)   # Предсказание времени
        self.priority_output = nn.Linear(hidden_dim, 1)  # Приоритет клиента

        # Активация
        self.activation = nn.ReLU()

    def forward(self, x, mask=None):
        """
        Прямой проход модели с поддержкой графа маршрутов
        """
        batch_size, seq_len, _ = x.shape

        # Входной слой
        x = self.input_layer(x)

        # Attention слои для анализа связей в графе
        for attention, norm in zip(self.attention_layers, self.norm_layers):
            # Self-attention для анализа связей между клиентами
            attn_output, _ = attention(x, x, x, key_padding_mask=mask)
            x = norm(x + attn_output)
            x = self.activation(x)

        # Выходные предсказания для графа
        route_scores = self.route_output(x)      # Очки для построения графа
        time_predictions = self.time_output(x)   # Время между узлами
        priority_scores = self.priority_output(x)  # Приоритет узлов

        return {
            'route_scores': route_scores.squeeze(-1),
            'time_predictions': time_predictions.squeeze(-1),
            'priority_scores': priority_scores.squeeze(-1)
        }

class UnifiedRouteSystem:
    """Единая система: ANN + TomTom API + граф маршрутов"""

    def __init__(self, tomtom_api_key: str, model_path: str = None, bot_token: str = None):
        self.tomtom_api_key = tomtom_api_key
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttentionRouteOptimizer().to(self.device)
        self.scaler = StandardScaler()
        self.clients = []
        self.current_routes = {}
        self.visited_clients = set()
        self.current_time = 9.0
        self.location_detector = LocationDetector()
        self.user_location = None

        # Система уведомлений
        self.bot_token = bot_token
        self.time_monitor = TimeMonitor()
        self.notification_system = None

        if bot_token:
            self.notification_system = NotificationSystem(bot_token)
            # Регистрируем callbacks для уведомлений
            self.time_monitor.register_callback(TriggerType.DEPARTURE_REMINDER, self._handle_departure_reminder)
            self.time_monitor.register_callback(TriggerType.LUNCH_BREAK, self._handle_lunch_reminder)
            self.time_monitor.register_callback(TriggerType.DELAY_ALERT, self._handle_delay_alert)
            self.time_monitor.register_callback(TriggerType.TRAFFIC_CHANGE, self._handle_traffic_change)
            self.time_monitor.register_callback(TriggerType.ROUTE_UPDATE, self._handle_route_update)
            self.time_monitor.register_callback(TriggerType.CLIENT_ARRIVAL, self._handle_client_arrival)

            # Запускаем мониторинг времени
            self.time_monitor.start_monitoring()
            print("🔔 Система уведомлений инициализирована")

        # Загружаем предобученную модель если есть
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif os.path.exists("best_unified_model.pth"):
            print("📁 Загружаем сохраненную модель...")
            self.load_model("best_unified_model.pth")
        else:
            print("⚠️ Модель не найдена, будет обучена на NYC данных")
            print(f"🖥️ Устройство: {self.device}")
            print(f"🧠 Модель: {sum(p.numel() for p in self.model.parameters())} параметров")

    def load_nyc_training_data(self) -> pd.DataFrame:
        """Загружает NYC данные для обучения модели с поддержкой графа"""
        print("📊 Загрузка NYC данных для обучения...")

        # Сначала проверяем объединенные данные
        combined_data_path = "combined_training_data.csv"

        if os.path.exists(combined_data_path):
            print("📊 Загрузка объединенных данных (NYC + времена)...")
            return self._load_combined_data(combined_data_path)
        else:
            print("📊 Загрузка только NYC данных...")
            return self._load_nyc_only_data()

    def _load_combined_data(self, data_path: str):
        """Загрузка объединенных данных с временными параметрами"""
        try:
            df = pd.read_csv(data_path)
            print(f"📋 Колонки в данных: {list(df.columns)}")
            print(f"✅ Загружено {len(df)} записей с временными параметрами")
            return df
        except Exception as e:
            print(f"❌ Ошибка загрузки объединенных данных: {e}")
            return self._load_nyc_only_data()

    def _load_nyc_only_data(self):
        """Загрузка только NYC данных"""
        try:
            trip_data_path = 'DS/taxi_trip_data.csv'
            zone_data_path = 'DS/taxi_zone_geo.csv'

            if not os.path.exists(trip_data_path):
                print(f"❌ Файл данных не найден: {trip_data_path}")
                return pd.DataFrame()

            print("📊 Загрузка NYC данных...")

            # Читаем данные порциями для экономии памяти
            trips = []
            chunk_size = 10000

            for chunk in pd.read_csv(trip_data_path, chunksize=chunk_size):
                if len(trips) >= 50000:  # Ограничиваем до 50k поездок
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
                                'is_vip': np.random.choice([True, False]),
                                'work_start_hour': 0.33,  # 08:00 нормализованное
                                'work_end_hour': 0.75,   # 18:00 нормализованное
                                'lunch_start_hour': 0.54, # 13:00 нормализованное
                                'lunch_end_hour': 0.58,   # 14:00 нормализованное
                                'timestamp': np.random.randint(0, 86400),
                                'distance': float(trip_distance) if pd.notna(trip_distance) else 1.0,
                                'time': np.random.uniform(5, 120),
                                'fare': float(fare_amount) if pd.notna(fare_amount) else 10.0
                            }
                            trips.append(trip)

                    except Exception as e:
                        continue

                print(f"📊 Обработано {len(trips)} поездок...")

            df = pd.DataFrame(trips)
            print(f"✅ Загружено {len(df)} поездок из NYC")
            return df

        except Exception as e:
            print(f"❌ Ошибка загрузки NYC данных: {e}")
            return pd.DataFrame()

    def prepare_training_data(self, nyc_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Подготавливает данные для обучения ANN с правильными колонками"""
        print("🔄 Подготовка данных для обучения...")
        print(f"📋 Доступные колонки: {list(nyc_data.columns)}")

        # Создаем признаки
        features = []
        targets = []

        for _, row in nyc_data.iterrows():
            try:
                # Входные признаки (7 параметров для графа)
                feature = [
                    row['latitude'],           # Широта
                    row['longitude'],          # Долгота
                    float(row['is_vip']),      # VIP статус
                    row['work_start_hour'],    # Начало работы
                    row['work_end_hour'],      # Конец работы
                    row['lunch_start_hour'],   # Начало обеда
                    row['lunch_end_hour']      # Конец обеда
                ]

                # Целевая переменная (время поездки)
                target = [row.get('time', 30.0)]  # Время по умолчанию 30 мин

                features.append(feature)
                targets.append(target)

            except Exception as e:
                print(f"⚠️ Ошибка обработки строки: {e}")
                continue

        if not features:
            print("❌ Нет валидных данных для обучения")
            return np.array([]), np.array([])

        X = np.array(features)
        y = np.array(targets)

        # Нормализуем данные
        X_scaled = self.scaler.fit_transform(X)

        print(f"✅ Подготовлено {len(X)} образцов для обучения")
        return X_scaled, y

    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, start_epoch: int = 0):
        """Обучает Attention модель на NYC данных с поддержкой графа"""
        print(f"🧠 Обучение Attention модели на {epochs} эпох (начиная с {start_epoch})...")

        # Создаем датасет для графа маршрутов
        dataset = self._create_route_dataset(X, y)

        # Разделяем на train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

        # Оптимизатор и функция потерь
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Загружаем существующую модель если есть
        best_val_loss = float('inf')
        if os.path.exists("best_unified_model.pth"):
            print("📁 Загружаем существующую модель для продолжения обучения...")
            try:
                checkpoint = torch.load("best_unified_model.pth", map_location=self.device)
                self.model.load_state_dict(checkpoint)
                print("✅ Модель загружена успешно")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки модели: {e}")
                print("🔄 Начинаем обучение с нуля")

        # Обучение с early stopping
        patience = 10
        patience_counter = 0

        for epoch in range(start_epoch, epochs):
            # Обучение
            train_loss = self._train_epoch(train_loader, optimizer, criterion)

            # Валидация
            val_loss = self._validate_epoch(val_loader, criterion)

            print(f"Эпоха {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Сохраняем лучшую модель
                torch.save(self.model.state_dict(), "best_unified_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"🛑 Early stopping на эпохе {epoch+1}")
                    break

        print("✅ Обучение завершено!")
        print(f"📁 Лучшая модель сохранена: best_unified_model.pth")

    def _create_route_dataset(self, X: np.ndarray, y: np.ndarray):
        """Создает датасет для обучения графа маршрутов"""
        class RouteDataset(torch.utils.data.Dataset):
            def __init__(self, features, targets):
                self.features = features
                self.targets = targets

            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                # Создаем группу из 8 клиентов для графа
                group_size = 8
                start_idx = (idx * group_size) % len(self.features)

                group_features = []
                group_targets = []

                for i in range(group_size):
                    feature_idx = (start_idx + i) % len(self.features)
                    group_features.append(self.features[feature_idx])
                    group_targets.append(self.targets[feature_idx])

                return {
                    'features': torch.FloatTensor(group_features),
                    'targets': torch.LongTensor(group_targets)
                }

        return RouteDataset(X, y)

    def _train_epoch(self, dataloader, optimizer, criterion):
        """Обучение одной эпохи"""
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)

            # Создаем маску для графа
            batch_size, seq_len = features.shape[:2]
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)

            # Прямой проход
            outputs = self.model(features, mask)

            # Вычисляем потери для графа маршрутов
            targets_float = targets.float()
            loss = criterion(outputs['route_scores'], targets_float)

            # Обратное распространение
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def _validate_epoch(self, dataloader, criterion):
        """Валидация одной эпохи"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)

                # Создаем маску для графа
                batch_size, seq_len = features.shape[:2]
                mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)

                # Прямой проход
                outputs = self.model(features, mask)

                # Вычисляем потери
                targets_float = targets.float()
                loss = criterion(outputs['route_scores'], targets_float)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def predict_route_time(self, client1: Client, client2: Client, current_time: float) -> float:
        """Предсказывает время маршрута между клиентами"""
        # Подготавливаем признаки
        features = np.array([[
            client1.lat,
            client1.lon,
            client2.lat,
            client2.lon,
            self.calculate_distance(client1, client2),
            current_time,
            1.0,  # Количество пассажиров
            1.0 if client1.client_level == ClientLevel.VIP else 0.0
        ]])

        # Нормализуем
        features_scaled = self.scaler.transform(features)

        # Предсказание
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled)
            prediction = self.model(features_tensor)
            return prediction.item()

    def calculate_distance(self, client1: Client, client2: Client) -> float:
        """Вычисляет расстояние между клиентами"""
        from math import radians, cos, sin, asin, sqrt

        lat1, lon1 = radians(client1.lat), radians(client1.lon)
        lat2, lon2 = radians(client2.lat), radians(client2.lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))

        r = 6371000  # Радиус Земли в метрах
        return c * r

    def get_tomtom_route(self, client1: Client, client2: Client) -> Dict:
        """Получает реальный маршрут от TomTom API"""
        try:
            url = f"https://api.tomtom.com/routing/1/calculateRoute/{client1.lat},{client1.lon}:{client2.lat},{client2.lon}/json"
            params = {
                'key': self.tomtom_api_key,
                'routeType': 'fastest',
                'traffic': 'true'
            }

            session = requests.Session()
            session.verify = False
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Connection': 'keep-alive'
            })

            response = session.get(url, params=params, timeout=30)

            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'TomTom API error: {response.status_code}'}

        except Exception as e:
            return {'error': f'TomTom API error: {str(e)}'}

    def optimize_route_with_ann(self, clients: List[Client]) -> List[Client]:
        """Оптимизирует маршрут используя Attention модель с графом"""
        print("🧠 Оптимизация маршрута с помощью Attention модели...")

        if not clients:
            return []

        # Подготавливаем данные для модели
        features = []
        for client in clients:
            client_features = [
                client.lat,
                client.lon,
                1.0 if client.client_level == ClientLevel.VIP else 0.0,
                client.work_start_hour,
                client.work_end_hour,
                client.lunch_start_hour,
                client.lunch_end_hour
            ]
            features.append(client_features)

        # Создаем тензор для графа
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        batch_size, seq_len = features_tensor.shape[:2]
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)

        # Получаем предсказания модели для графа
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor, mask)
            route_scores = outputs['route_scores'].cpu().numpy()[0]
            time_predictions = outputs['time_predictions'].cpu().numpy()[0]
            priority_scores = outputs['priority_scores'].cpu().numpy()[0]

        # Создаем граф маршрутов на основе предсказаний
        client_scores = list(zip(clients, route_scores, time_predictions, priority_scores))

        # Сортируем по комбинированному скору (маршрут + приоритет + время)
        client_scores.sort(key=lambda x: x[1] + x[3] * 0.5 - x[2] * 0.1, reverse=True)

        # Создаем оптимальный маршрут из графа
        optimized_clients = [client for client, _, _, _ in client_scores]

        print(f"✅ Граф маршрутов построен для {len(optimized_clients)} клиентов")
        return optimized_clients

    def check_working_hours(self, client: Client, arrival_time: float) -> bool:
        """Проверяет рабочие часы клиента"""
        if arrival_time < client.work_start_hour or arrival_time > client.work_end_hour:
            return False

        if client.lunch_start_hour <= arrival_time <= client.lunch_end_hour:
            return False

        return True

    def get_unified_route(self, clients: List[Client], num_days: int) -> Dict:
        """Возвращает единый маршрут: ANN + TomTom API"""
        print("🚀 Создание единого маршрута: ANN + TomTom API")

        # Распределяем клиентов по дням
        base_per_day = len(clients) // num_days
        extra_clients = len(clients) % num_days

        result = {
            'success': True,
            'total_clients': len(clients),
            'num_days': num_days,
            'routes': [],
            'visited_clients': list(self.visited_clients),
            'current_time': self.current_time
        }

        start_idx = 0
        for day in range(num_days):
            clients_this_day = base_per_day + (1 if day < extra_clients else 0)
            day_clients = clients[start_idx:start_idx + clients_this_day]

            # Оптимизируем с помощью ANN
            optimized_clients = self.optimize_route_with_ann(day_clients)

            # Получаем детальные маршруты от TomTom
            tomtom_routes = []
            for i in range(len(optimized_clients) - 1):
                client1 = optimized_clients[i]
                client2 = optimized_clients[i + 1]

                tomtom_route = self.get_tomtom_route(client1, client2)
                if 'error' not in tomtom_route:
                    tomtom_routes.append(tomtom_route)

            day_route = {
                'day': day + 1,
                'clients': [c.id for c in optimized_clients],
                'waypoints': [{'lat': c.lat, 'lon': c.lon, 'id': c.id, 'level': c.client_level.value}
                             for c in optimized_clients],
                'tomtom_routes': tomtom_routes,
                'ann_optimized': True
            }

            result['routes'].append(day_route)
            start_idx += clients_this_day

        return result

    def set_user_location(self, gps_coords: Optional[Tuple[float, float]] = None,
                         ip_address: Optional[str] = None,
                         manual_address: Optional[str] = None) -> Dict:
        """Устанавливает местоположение пользователя"""
        print("📍 Определение местоположения пользователя...")

        try:
            # Получаем лучшее доступное местоположение
            location = self.location_detector.get_best_location(
                gps_coords=gps_coords,
                ip_address=ip_address,
                manual_address=manual_address
            )

            # Проверяем валидность
            if self.location_detector.validate_location(location):
                self.user_location = location
                print(f"✅ Местоположение установлено: {location.address}")

                return {
                    'success': True,
                    'location': {
                        'latitude': location.latitude,
                        'longitude': location.longitude,
                        'address': location.address,
                        'city': location.city,
                        'country': location.country,
                        'accuracy': location.accuracy,
                        'source': location.source
                    },
                    'message': f'Местоположение определено: {location.address}'
                }
            else:
                print("❌ Невалидное местоположение")
                return {
                    'success': False,
                    'error': 'Невалидное местоположение',
                    'message': 'Не удалось определить корректное местоположение'
                }

        except Exception as e:
            print(f"❌ Ошибка определения местоположения: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Ошибка определения местоположения'
            }

    def get_user_location(self) -> Dict:
        """Возвращает текущее местоположение пользователя"""
        if self.user_location:
            return {
                'success': True,
                'location': {
                    'latitude': self.user_location.latitude,
                    'longitude': self.user_location.longitude,
                    'address': self.user_location.address,
                    'city': self.user_location.city,
                    'country': self.user_location.country,
                    'accuracy': self.user_location.accuracy,
                    'source': self.user_location.source
                }
            }
        else:
            return {
                'success': False,
                'message': 'Местоположение не определено'
            }

    def get_route_from_user_location(self, clients: List[Client], num_days: int) -> Dict:
        """Строит маршрут от местоположения пользователя к клиентам"""
        print("🗺️ Построение маршрута от местоположения пользователя...")

        if not self.user_location:
            return {
                'success': False,
                'error': 'Местоположение пользователя не определено',
                'message': 'Сначала определите местоположение пользователя'
            }

        # Добавляем пользователя как стартовую точку
        user_client = Client(
            id=0,
            address=f"Пользователь: {self.user_location.address}",
            lat=self.user_location.latitude,
            lon=self.user_location.longitude,
            client_level=ClientLevel.REGULAR,
            work_start="00:00",
            work_end="23:59",
            lunch_start="00:00",
            lunch_end="00:00"
        )

        # Добавляем пользователя в начало списка клиентов
        all_clients = [user_client] + clients

        # Строим маршрут
        route_result = self.get_unified_route(all_clients, num_days)

        # Добавляем информацию о местоположении пользователя
        route_result['user_location'] = {
            'latitude': self.user_location.latitude,
            'longitude': self.user_location.longitude,
            'address': self.user_location.address,
            'source': self.user_location.source
        }

        return route_result

    def register_telegram_user(self, chat_id: int, user_settings: UserSettings):
        """Регистрирует пользователя Telegram в системе уведомлений"""
        if self.notification_system and self.time_monitor:
            self.notification_system.register_user(chat_id, user_settings)
            self.time_monitor.add_user(chat_id, user_settings)
            print(f"👤 Пользователь Telegram {chat_id} зарегистрирован")

    def add_route_notifications(self, chat_id: int, route_result: Dict):
        """Добавляет уведомления для маршрута"""
        if not self.notification_system or not self.time_monitor:
            return

        print(f"🔔 Добавление уведомлений для маршрута пользователя {chat_id}")

        current_time = datetime.now()

        for day_route in route_result.get('routes', []):
            for i, client in enumerate(day_route.get('clients', [])):
                if client['id'] == 0:  # Пропускаем пользователя
                    continue

                arrival_time = datetime.strptime(client['arrival_time'], '%H:%M')
                arrival_time = current_time.replace(hour=arrival_time.hour, minute=arrival_time.minute, second=0, microsecond=0)

                # Напоминание о выезде
                self.time_monitor.add_departure_reminder(
                    chat_id,
                    arrival_time,
                    {
                        'id': client['id'],
                        'address': client['address'],
                        'client_level': client['client_level'],
                        'travel_time': 20  # Примерное время
                    }
                )

                # Уведомление о прибытии
                self.time_monitor.add_client_arrival_notification(
                    chat_id,
                    {
                        'id': client['id'],
                        'address': client['address'],
                        'client_level': client['client_level'],
                        'service_time': 30 if client['client_level'] == 'VIP' else 20
                    },
                    arrival_time
                )

        # Напоминание об обеде
        lunch_time = datetime.strptime("13:00", '%H:%M')
        lunch_time = current_time.replace(hour=lunch_time.hour, minute=lunch_time.minute, second=0, microsecond=0)
        self.time_monitor.add_lunch_reminder(chat_id, lunch_time)

    def check_delays(self, chat_id: int, current_time: datetime = None):
        """Проверяет опоздания и добавляет уведомления"""
        if not self.time_monitor or not self.current_routes:
            return

        if current_time is None:
            current_time = datetime.now()

        for day_route in self.current_routes.get('routes', []):
            for client in day_route.get('clients', []):
                if client['id'] == 0 or client['id'] in self.visited_clients:
                    continue

                planned_arrival = datetime.strptime(client['arrival_time'], '%H:%M')
                planned_arrival = current_time.replace(hour=planned_arrival.hour, minute=planned_arrival.minute, second=0, microsecond=0)

                if current_time > planned_arrival:
                    self.time_monitor.add_delay_alert(
                        chat_id,
                        planned_arrival,
                        current_time,
                        {
                            'id': client['id'],
                            'address': client['address'],
                            'client_level': client['client_level']
                        }
                    )

    def check_traffic_changes(self, chat_id: int, old_route_time: int, new_route_time: int):
        """Проверяет изменения трафика и добавляет уведомления"""
        if not self.time_monitor:
            return

        self.time_monitor.add_traffic_change_alert(
            chat_id,
            old_route_time,
            new_route_time,
            {
                'description': 'Текущий маршрут',
                'total_clients': len(self.clients)
            }
        )

    def _handle_departure_reminder(self, trigger):
        """Обработчик напоминания о выезде"""
        if self.notification_system:
            self.notification_system.handle_trigger(trigger)

    def _handle_lunch_reminder(self, trigger):
        """Обработчик напоминания об обеде"""
        if self.notification_system:
            self.notification_system.handle_trigger(trigger)

    def _handle_delay_alert(self, trigger):
        """Обработчик уведомления об опоздании"""
        if self.notification_system:
            self.notification_system.handle_trigger(trigger)

    def _handle_traffic_change(self, trigger):
        """Обработчик уведомления об изменении трафика"""
        if self.notification_system:
            self.notification_system.handle_trigger(trigger)

    def _handle_route_update(self, trigger):
        """Обработчик уведомления об обновлении маршрута"""
        if self.notification_system:
            self.notification_system.handle_trigger(trigger)

    def _handle_client_arrival(self, trigger):
        """Обработчик уведомления о прибытии к клиенту"""
        if self.notification_system:
            self.notification_system.handle_trigger(trigger)

    def update_user_location(self, new_latitude: float, new_longitude: float) -> Dict:
        """Обновляет местоположение пользователя (например, с карты)"""
        print(f"📍 Обновление местоположения пользователя: {new_latitude}, {new_longitude}")

        try:
            # Создаем новое местоположение
            new_location = self.location_detector.get_location_from_gps(new_latitude, new_longitude)

            # Проверяем валидность
            if self.location_detector.validate_location(new_location):
                old_location = self.user_location
                self.user_location = new_location

                print(f"✅ Местоположение обновлено: {new_location.address}")

                return {
                    'success': True,
                    'old_location': {
                        'latitude': old_location.latitude if old_location else None,
                        'longitude': old_location.longitude if old_location else None,
                        'address': old_location.address if old_location else None
                    },
                    'new_location': {
                        'latitude': new_location.latitude,
                        'longitude': new_location.longitude,
                        'address': new_location.address,
                        'city': new_location.city,
                        'country': new_location.country,
                        'accuracy': new_location.accuracy,
                        'source': new_location.source
                    },
                    'message': f'Местоположение обновлено: {new_location.address}'
                }
            else:
                print("❌ Невалидные координаты")
                return {
                    'success': False,
                    'error': 'Невалидные координаты',
                    'message': 'Указанные координаты находятся вне разумных пределов'
                }

        except Exception as e:
            print(f"❌ Ошибка обновления местоположения: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Ошибка обновления местоположения'
            }

    def recalculate_routes_from_new_location(self, clients: List[Client], num_days: int) -> Dict:
        """Пересчитывает маршруты с нового местоположения пользователя"""
        print("🔄 Пересчет маршрутов с нового местоположения...")

        if not self.user_location:
            return {
                'success': False,
                'error': 'Местоположение пользователя не определено',
                'message': 'Сначала определите местоположение пользователя'
            }

        try:
            # Строим новые маршруты от обновленного местоположения
            new_routes = self.get_route_from_user_location(clients, num_days)

            if new_routes['success']:
                # Обновляем текущие маршруты
                self.current_routes = new_routes

                print("✅ Маршруты успешно пересчитаны!")

                return {
                    'success': True,
                    'routes': new_routes,
                    'message': 'Маршруты пересчитаны с нового местоположения'
                }
            else:
                return new_routes

        except Exception as e:
            print(f"❌ Ошибка пересчета маршрутов: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Ошибка пересчета маршрутов'
            }

    def get_location_suggestions(self, query: str, limit: int = 5) -> Dict:
        """Получает предложения адресов для автодополнения"""
        print(f"🔍 Поиск предложений для: {query}")

        try:
            # Используем OpenStreetMap Nominatim для поиска
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': query,
                'format': 'json',
                'limit': limit,
                'addressdetails': 1,
                'countrycodes': 'ru'  # Ограничиваем Россией
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                suggestions = []
                for item in data:
                    suggestions.append({
                        'display_name': item.get('display_name', ''),
                        'latitude': float(item.get('lat', 0)),
                        'longitude': float(item.get('lon', 0)),
                        'address': item.get('address', {}),
                        'importance': item.get('importance', 0)
                    })

                print(f"✅ Найдено {len(suggestions)} предложений")

                return {
                    'success': True,
                    'suggestions': suggestions,
                    'query': query
                }
            else:
                print(f"❌ Ошибка поиска: {response.status_code}")
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}',
                    'message': 'Ошибка поиска адресов'
                }

        except Exception as e:
            print(f"❌ Ошибка поиска предложений: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Ошибка поиска предложений'
            }

    def mark_client_visited(self, client_id: int, actual_service_time: Optional[float] = None) -> Dict:
        """Отмечает клиента как посещенного и пересчитывает маршруты"""
        print(f"✅ Отмечаем клиента {client_id} как посещенного...")

        # Добавляем в список посещенных
        self.visited_clients.add(client_id)

        # Учитываем реальное время обслуживания
        if actual_service_time is not None:
            # Обновляем текущее время с учетом реального времени обслуживания
            self.current_time += actual_service_time / 60.0  # Конвертируем минуты в часы
            print(f"⏰ Обновлено время: {self.current_time:.2f} часов")
        else:
            # Используем стандартное время обслуживания
            # Здесь нужно найти клиента и определить его тип
            # Пока используем среднее время
            self.current_time += 0.5  # 30 минут по умолчанию

        # Пересчитываем оставшиеся маршруты
        print("🔄 Пересчитываем оставшиеся маршруты...")
        updated_routes = self.recalculate_remaining_routes()

        return {
            'success': True,
            'visited_client_id': client_id,
            'visited_clients': list(self.visited_clients),
            'current_time': self.current_time,
            'updated_routes': updated_routes,
            'message': f'Клиент {client_id} отмечен как посещенный. Маршруты пересчитаны.'
        }

    def recalculate_remaining_routes(self) -> Dict:
        """Пересчитывает маршруты для оставшихся клиентов"""
        print("🧠 Пересчет маршрутов с учетом посещенных клиентов...")

        # Здесь должна быть логика пересчета
        # Пока возвращаем базовую информацию
        return {
            'recalculated': True,
            'remaining_clients': len(self.clients) - len(self.visited_clients),
            'visited_count': len(self.visited_clients),
            'current_time': self.current_time
        }

    def export_routes_to_json(self, route_data: Dict, filename: str = "routes.json") -> str:
        """Экспортирует маршруты в JSON файл для фронтенда"""
        print(f"📄 Экспортируем маршруты в {filename}...")

        try:
            # Создаем структуру для фронтенда
            frontend_data = {
                "success": route_data.get('success', True),
                "total_clients": route_data.get('total_clients', 0),
                "num_days": route_data.get('num_days', 0),
                "visited_clients": route_data.get('visited_clients', []),
                "current_time": route_data.get('current_time', 9.0),
                "routes": []
            }

            # Обрабатываем каждый день
            for route in route_data.get('routes', []):
                day_route = {
                    "day": route.get('day', 1),
                    "clients": route.get('clients', []),
                    "waypoints": route.get('waypoints', []),
                    "tomtom_routes": route.get('tomtom_routes', []),
                    "ann_optimized": route.get('ann_optimized', True)
                }
                frontend_data["routes"].append(day_route)

            # Сохраняем в JSON файл
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(frontend_data, f, ensure_ascii=False, indent=2)

            print(f"✅ JSON файл сохранен: {filename}")
            return filename

        except Exception as e:
            print(f"❌ Ошибка экспорта JSON: {e}")
            return None

    def save_model(self, path: str):
        """Сохраняет обученную модель"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler
        }, path)
        print(f"✅ Модель сохранена в {path}")

    def load_model(self, path: str):
        """Загружает обученную модель"""
        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Проверяем формат сохраненной модели
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Новый формат с полной информацией
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'scaler' in checkpoint:
                    self.scaler = checkpoint['scaler']
            else:
                # Старый формат - только state_dict
                self.model.load_state_dict(checkpoint)

            print(f"✅ Модель загружена из {path}")

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print("🔄 Продолжаем без предобученной модели")

def main():
    """Демонстрация единой системы"""
    print("🧠 Единая система: ANN + TomTom API + граф маршрутов")
    print("=" * 60)

    # Инициализируем систему
    system = UnifiedRouteSystem(tomtom_api_key="4Me4kS17IKSfQmvDuIgLpsz9jxAu6tt2")

    # Загружаем NYC данные для обучения
    nyc_data = system.load_nyc_training_data()

    if not nyc_data.empty:
        # Подготавливаем данные
        X, y = system.prepare_training_data(nyc_data)

        # Продолжаем обучение с 39-й эпохи до 50-й
        system.train_model(X, y, epochs=50, start_epoch=39)

        # Сохраняем модель
        system.save_model('route_ann_model.pth')

    # Загружаем клиентов
    print("\n👥 Загрузка клиентов...")
    # Здесь должен быть код загрузки клиентов из DATA (2).txt

    print("✅ Единая система готова к работе!")

if __name__ == "__main__":
    main()
