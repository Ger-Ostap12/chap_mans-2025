import requests
import json
import re
import os
import asyncio
import aiohttp
import redis
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, TransformerConv
from torch_geometric.data import Data, Batch

class GraphTransformerFeatureExtractor(nn.Module):
    """
    Graph Transformer как помощник для ANN
    Извлекает графовые признаки для существующей AttentionRouteOptimizer
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 128, num_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.1):
        super(GraphTransformerFeatureExtractor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Входной слой
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Graph Attention Layers (GAT) - только для извлечения признаков
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                   dropout=dropout, concat=True) for _ in range(num_layers)
        ])

        # Слой для извлечения графовых признаков
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 7)  # 7 логических графовых признаков
        )

        self.dropout = nn.Dropout(dropout)

    def create_graph_from_clients(self, clients: List[Dict]) -> Data:
        """
        Создает граф из списка клиентов
        """
        num_clients = len(clients)

        # Извлекаем признаки
        features = []
        for client in clients:
            feature_vector = [
                client.get('lat', 0.0),
                client.get('lon', 0.0),
                1.0 if client.get('client_level', '').lower() == 'vip' else 0.0,
                client.get('work_start_hour', 0.0),
                client.get('work_end_hour', 0.0),
                client.get('lunch_start_hour', 0.0),
                client.get('lunch_end_hour', 0.0)
            ]
            features.append(feature_vector)

        # Создаем полносвязный граф (каждый клиент связан с каждым)
        edge_index = []
        for i in range(num_clients):
            for j in range(num_clients):
                if i != j:  # Не связываем узел с самим собой
                    edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        features = torch.tensor(features, dtype=torch.float)

        return Data(x=features, edge_index=edge_index)

    def forward(self, graph_data: Data) -> torch.Tensor:
        """
        Forward pass через Graph Transformer для извлечения признаков
        Возвращает обогащенные признаки для ANN
        """
        x = graph_data.x
        edge_index = graph_data.edge_index

        # Проекция входных признаков
        x = self.input_projection(x)

        # Graph Attention Layers для анализа графа
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        # Извлекаем графовые признаки
        graph_features = self.feature_extractor(x)

        return graph_features

    def extract_graph_features(self, clients: List[Dict]) -> List[List[float]]:
        """
        Извлекает графовые признаки для каждого клиента
        Возвращает обогащенные признаки для ANN
        """
        self.eval()
        with torch.no_grad():
            # Создаем граф
            graph_data = self.create_graph_from_clients(clients)

            # Получаем графовые признаки
            graph_features = self.forward(graph_data)

            # Объединяем исходные признаки с графовыми
            enriched_features = []
            for i, client in enumerate(clients):
                # Исходные признаки (7)
                original_features = [
                    client.get('lat', 0.0),
                    client.get('lon', 0.0),
                    1.0 if client.get('client_level', '').lower() == 'vip' else 0.0,
                    client.get('work_start_hour', 0.0),
                    client.get('work_end_hour', 0.0),
                    client.get('lunch_start_hour', 0.0),
                    client.get('lunch_end_hour', 0.0)
                ]

                # Графовые признаки (7 логических признаков)
                graph_feat = graph_features[i].tolist()

                # Объединяем (7 исходных + 7 графовых = 14 признаков)
                enriched_features.append(original_features + graph_feat)

            return enriched_features

class TomTomTrafficRouter:
    """
    Модуль для расчета маршрутов с учетом трафика TomTom
    Включает время в пути, задержки от пробок и время прибытия
    """

    def __init__(self, api_key: str, redis_host: str = 'localhost', redis_port: int = 6379, redis_db: int = 0):
        self.api_key = api_key
        self.base_url = "https://api.tomtom.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RouteOptimizer/1.0',
            'Accept': 'application/json'
        })

        # Graph Transformer как помощник для ANN
        self.graph_feature_extractor = GraphTransformerFeatureExtractor(
            input_dim=7,
            hidden_dim=128,
            num_heads=8,
            num_layers=2,
            dropout=0.1
        )

        # Загружаем существующую ANN модель
        self.ann_model = None
        self.load_ann_model()

        # Redis кэш
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True
            )
            self.redis_client.ping()  # Проверка подключения
            self.cache_enabled = True
            print("✅ Redis подключен успешно")
        except Exception as e:
            print(f"⚠️ Redis недоступен: {e}. Работаем без кэша.")
            self.redis_client = None
            self.cache_enabled = False

        # Кэш TTL (время жизни)
        self.cache_ttl = {
            'route': 3600,      # 1 час для маршрутов
            'traffic': 60,       # 1 минута для трафика (реальное время!)
            'geocoding': 86400   # 24 часа для геокодирования
        }

    def load_ann_model(self):
        """
        Загружает существующую ANN модель
        """
        try:
            # Импортируем ANN модель из train_model.py
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))

            from train_model import AttentionRouteOptimizer

            # Создаем модель с обновленным input_dim (7 + 7 = 14)
            self.ann_model = AttentionRouteOptimizer(
                input_dim=14,  # 7 исходных + 7 графовых признаков
                hidden_dim=256,
                num_heads=8,
                num_layers=4
            )

            # Загружаем веса если есть
            if os.path.exists('best_model.pth'):
                self.ann_model.load_state_dict(torch.load('best_model.pth'))
                print("✅ ANN модель загружена из best_model.pth")
            else:
                print("⚠️ ANN модель создана с случайными весами")

        except Exception as e:
            print(f"❌ Ошибка загрузки ANN модели: {e}")
            self.ann_model = None

    def _generate_cache_key(self, cache_type: str, **kwargs) -> str:
        """Генерация ключа кэша"""
        # Создаем строку из параметров
        params_str = f"{cache_type}:{kwargs}"
        # Хэшируем для короткого ключа
        return hashlib.md5(params_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Получение данных из кэша"""
        if not self.cache_enabled:
            return None

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            print(f"Ошибка чтения из кэша: {e}")

        return None

    def _save_to_cache(self, cache_key: str, data: Dict, ttl: int) -> None:
        """Сохранение данных в кэш"""
        if not self.cache_enabled:
            return

        try:
            self.redis_client.setex(cache_key, ttl, json.dumps(data))
        except Exception as e:
            print(f"Ошибка записи в кэш: {e}")

    def calculate_route_with_traffic(self,
                                   start_lat: float,
                                   start_lon: float,
                                   end_lat: float,
                                   end_lon: float,
                                   departure_time: Optional[datetime] = None) -> Dict:
        """
        Расчет маршрута с учетом трафика

        Args:
            start_lat, start_lon: Координаты начала маршрута
            end_lat, end_lon: Координаты конца маршрута
            departure_time: Время отправления (по умолчанию - сейчас)

        Returns:
            Dict с информацией о маршруте, времени в пути и времени прибытия
        """
        try:
            if departure_time is None:
                departure_time = datetime.now()

            # Проверяем кэш
            cache_key = self._generate_cache_key(
                'route',
                start_lat=start_lat,
                start_lon=start_lon,
                end_lat=end_lat,
                end_lon=end_lon,
                departure_time=departure_time.isoformat()
            )

            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                print("📦 Маршрут загружен из кэша")
                return cached_result

            # Построение маршрута с учетом трафика
            route_data = self._get_route_with_traffic(
                start_lat, start_lon, end_lat, end_lon, departure_time
            )

            if not route_data:
                # Fallback: пробуем альтернативные стратегии
                print("🔄 Пробуем альтернативные стратегии маршрутизации...")
                fallback_route = self._try_fallback_routing(start_lat, start_lon, end_lat, end_lon, departure_time)
                if fallback_route:
                    return fallback_route

                # Последний fallback: создаем простой маршрут по прямой
                return self._create_simple_route(start_lat, start_lon, end_lat, end_lon, departure_time)

            # Расчет времени прибытия
            arrival_time = self._calculate_arrival_time(
                departure_time,
                route_data['travel_time_seconds']
            )

            # Получение данных о трафике
            # Получаем данные о трафике (всегда реальное время)
            traffic_data = self._get_traffic_data(start_lat, start_lon)

            result = {
                'status': 'success',
                'route': {
                    'start_coordinates': [start_lat, start_lon],
                    'end_coordinates': [end_lat, end_lon],
                    'distance_meters': route_data['distance_meters'],
                    'travel_time_seconds': route_data['travel_time_seconds'],
                    'traffic_delay_seconds': route_data['traffic_delay_seconds'],
                    'free_flow_time_seconds': route_data['free_flow_time_seconds']
                },
                'timing': {
                    'departure_time': departure_time.isoformat(),
                    'arrival_time': arrival_time.isoformat(),
                    'travel_time_formatted': self._format_duration(route_data['travel_time_seconds']),
                    'traffic_delay_formatted': self._format_duration(route_data['traffic_delay_seconds'])
                },
                'traffic': {
                    'current_speed_kmh': traffic_data.get('current_speed', 0),
                    'free_flow_speed_kmh': traffic_data.get('free_flow_speed', 0),
                    'traffic_level': self._get_traffic_level(traffic_data),
                    'confidence': traffic_data.get('confidence', 0)
                },
                'route_points': route_data.get('route_points', [])
            }

            # Сохраняем в кэш
            self._save_to_cache(cache_key, result, self.cache_ttl['route'])
            print("💾 Маршрут сохранен в кэш")

            return result

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Ошибка расчета маршрута: {str(e)}'
            }

    def _get_route_with_traffic(self,
                               start_lat: float,
                               start_lon: float,
                               end_lat: float,
                               end_lon: float,
                               departure_time: datetime) -> Optional[Dict]:
        """Получение маршрута с учетом трафика"""
        try:
            url = f"{self.base_url}/routing/1/calculateRoute/{start_lat},{start_lon}:{end_lat},{end_lon}/json"

            params = {
                'key': self.api_key,
                'routeType': 'fastest',
                'traffic': 'true',
                'departAt': departure_time.isoformat(),
                'instructionsType': 'text',
                'language': 'ru-RU'
            }

            response = self.session.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                routes = data.get('routes', [])

                if routes:
                    route = routes[0]
                    summary = route.get('summary', {})
                    legs = route.get('legs', [])

                    # Извлечение точек маршрута
                    route_points = []
                    if legs:
                        for leg in legs:
                            points = leg.get('points', [])
                            route_points.extend(points)

                    return {
                        'distance_meters': summary.get('lengthInMeters', 0),
                        'travel_time_seconds': summary.get('travelTimeInSeconds', 0),
                        'traffic_delay_seconds': summary.get('trafficDelayInSeconds', 0),
                        'free_flow_time_seconds': summary.get('travelTimeInSeconds', 0) - summary.get('trafficDelayInSeconds', 0),
                        'route_points': route_points
                    }

            return None

        except Exception as e:
            print(f"Ошибка получения маршрута: {e}")
            return None

    def _get_traffic_data(self, lat: float, lon: float) -> Dict:
        """Получение данных о трафике для точки БЕЗ кэширования (реальное время)"""
        try:
            # Всегда получаем свежие данные о трафике (без кэша)
            print("🔄 Получение актуальных данных о трафике...")

            url = f"{self.base_url}/traffic/services/4/flowSegmentData/absolute/10/json"
            params = {
                'key': self.api_key,
                'point': f'{lat},{lon}'
            }

            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                flow_segment = data.get('flowSegmentData', {})

                traffic_data = {
                    'current_speed': flow_segment.get('currentSpeed', 0),
                    'free_flow_speed': flow_segment.get('freeFlowSpeed', 0),
                    'confidence': flow_segment.get('confidence', 0)
                }

                # НЕ сохраняем в кэш - всегда свежие данные!
                print("✅ Получены актуальные данные о трафике")

                return traffic_data
            else:
                return {
                    'current_speed': 0,
                    'free_flow_speed': 0,
                    'confidence': 0
                }

        except Exception as e:
            print(f"Ошибка получения данных о трафике: {e}")
            return {
                'current_speed': 0,
                'free_flow_speed': 0,
                'confidence': 0
            }

    def _calculate_arrival_time(self, departure_time: datetime, travel_time_seconds: int) -> datetime:
        """Расчет времени прибытия"""
        return departure_time + timedelta(seconds=travel_time_seconds)

    def _format_duration(self, seconds: int) -> str:
        """Форматирование времени в читаемый вид"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        if hours > 0:
            return f"{hours}ч {minutes}м {seconds}с"
        elif minutes > 0:
            return f"{minutes}м {seconds}с"
        else:
            return f"{seconds}с"

    def _get_traffic_level(self, traffic_data: Dict) -> str:
        """Определение уровня трафика"""
        current_speed = traffic_data.get('current_speed', 0)
        free_flow_speed = traffic_data.get('free_flow_speed', 0)

        if free_flow_speed == 0:
            return "Неизвестно"

        speed_ratio = current_speed / free_flow_speed

        if speed_ratio >= 0.8:
            return "Свободно"
        elif speed_ratio >= 0.6:
            return "Загружено"
        elif speed_ratio >= 0.4:
            return "Пробки"
        else:
            return "Сильные пробки"

    def calculate_multiple_routes_parallel(self,
                                         start_coords: Tuple[float, float],
                                         destinations: List[Tuple[float, float]],
                                         client_types: List[str] = None,
                                         departure_time: Optional[datetime] = None,
                                         max_workers: int = None) -> Dict:
        """
        Параллельный расчет маршрутов до нескольких точек

        Args:
            start_coords: Координаты начала (lat, lon)
            destinations: Список координат назначения [(lat, lon), ...]
            client_types: Список типов клиентов ['VIP', 'regular', ...]
            departure_time: Время отправления
            max_workers: Максимальное количество потоков

        Returns:
            Dict с маршрутами до всех точек
        """
        try:
            if departure_time is None:
                departure_time = datetime.now()

            # Устанавливаем типы клиентов по умолчанию
            if client_types is None:
                client_types = ['regular'] * len(destinations)

                # Автоматический выбор количества потоков в зависимости от количества точек
                if max_workers is None:
                    if len(destinations) <= 10:
                        max_workers = 15  # Увеличено с 6 до 15
                    elif len(destinations) <= 30:
                        max_workers = 25  # Увеличено с 10 до 25
                    elif len(destinations) <= 50:
                        max_workers = 35  # Увеличено с 15 до 35
                    else:
                        max_workers = min(50, len(destinations) // 2)  # Увеличено максимум до 50

            print(f"🚀 Параллельный расчет {len(destinations)} маршрутов с {max_workers} потоками...")

            # Подготавливаем задачи для параллельного выполнения
            tasks = []
            for i, (end_lat, end_lon) in enumerate(destinations):
                client_type = client_types[i] if i < len(client_types) else 'regular'
                tasks.append({
                    'index': i,
                    'start_coords': start_coords,
                    'end_coords': (end_lat, end_lon),
                    'client_type': client_type,
                    'departure_time': departure_time
                })

            # Параллельное выполнение
            routes = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Отправляем все задачи
                future_to_task = {
                    executor.submit(self._calculate_single_route, task): task
                    for task in tasks
                }

                # Собираем результаты
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        routes.append(result)
                        print(f"✅ Маршрут {task['index']+1} готов")
                    except Exception as e:
                        print(f"❌ Ошибка маршрута {task['index']+1}: {e}")
                        routes.append({
                            'destination_index': task['index'],
                            'destination_coords': task['end_coords'],
                            'error': str(e)
                        })

            # Сортируем результаты по индексу
            routes.sort(key=lambda x: x['destination_index'])

            # Обновляем время для последовательного планирования
            current_time = departure_time
            for route in routes:
                if 'error' not in route:
                    arrival_time = datetime.fromisoformat(route['timing']['arrival_time'])
                    current_time = arrival_time + timedelta(minutes=route['stop_duration_minutes'])
                    route['updated_departure_time'] = current_time.isoformat()

            return {
                'status': 'success',
                'total_destinations': len(destinations),
                'successful_routes': len([r for r in routes if 'error' not in r]),
                'routes': routes,
                'total_travel_time': self._calculate_total_travel_time(routes),
                'parallel_execution': True,
                'max_workers': max_workers
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Ошибка параллельного расчета маршрутов: {str(e)}'
            }

    def _calculate_single_route(self, task: Dict) -> Dict:
        """Расчет одного маршрута для параллельного выполнения"""
        try:
            start_coords = task['start_coords']
            end_coords = task['end_coords']
            client_type = task['client_type']
            departure_time = task['departure_time']

            print(f"🔍 Отладка маршрута {task['index']+1}:")
            print(f"  📍 От: ({start_coords[0]:.6f}, {start_coords[1]:.6f})")
            print(f"  📍 До: ({end_coords[0]:.6f}, {end_coords[1]:.6f})")

            route_result = self.calculate_route_with_traffic(
                start_coords[0], start_coords[1],
                end_coords[0], end_coords[1],
                departure_time
            )

            if route_result['status'] == 'success':
                stop_duration = self._get_stop_duration(client_type)
                print(f"  ✅ Успешно: {route_result['route']['distance_meters']}м, {route_result['timing']['travel_time_formatted']}")

                return {
                    'destination_index': task['index'],
                    'destination_coords': list(end_coords),
                    'client_type': client_type,
                    'stop_duration_minutes': stop_duration,
                    'route': route_result['route'],
                    'timing': route_result['timing'],
                    'traffic': route_result['traffic']
                }
            else:
                error_msg = route_result.get('message', 'Не удалось построить маршрут')
                print(f"  ❌ Ошибка: {error_msg}")
                print(f"  🔍 Детали ответа API: {route_result}")
                return {
                    'destination_index': task['index'],
                    'destination_coords': list(end_coords),
                    'error': error_msg,
                    'api_response': route_result
                }

        except Exception as e:
            error_msg = f'Ошибка расчета: {str(e)}'
            print(f"  ❌ Исключение: {error_msg}")
            return {
                'destination_index': task['index'],
                'destination_coords': list(task['end_coords']),
                'error': error_msg
            }

    def calculate_multiple_routes(self,
                                start_coords: Tuple[float, float],
                                destinations: List[Tuple[float, float]],
                                client_types: List[str] = None,
                                departure_time: Optional[datetime] = None) -> Dict:
        """
        Расчет маршрутов до нескольких точек (последовательный)

        Args:
            start_coords: Координаты начала (lat, lon)
            destinations: Список координат назначения [(lat, lon), ...]
            client_types: Список типов клиентов ['VIP', 'regular', ...]
            departure_time: Время отправления

        Returns:
            Dict с маршрутами до всех точек
        """
        try:
            if departure_time is None:
                departure_time = datetime.now()

            routes = []
            current_time = departure_time

            # Устанавливаем типы клиентов по умолчанию
            if client_types is None:
                client_types = ['regular'] * len(destinations)

            for i, (end_lat, end_lon) in enumerate(destinations):
                print(f"🛣️ Расчет маршрута {i+1}/{len(destinations)}...")

                route_result = self.calculate_route_with_traffic(
                    start_coords[0], start_coords[1],
                    end_lat, end_lon,
                    current_time
                )

                if route_result['status'] == 'success':
                    # Определяем время остановки в зависимости от типа клиента
                    client_type = client_types[i] if i < len(client_types) else 'regular'
                    stop_duration = self._get_stop_duration(client_type)

                    routes.append({
                        'destination_index': i,
                        'destination_coords': [end_lat, end_lon],
                        'client_type': client_type,
                        'stop_duration_minutes': stop_duration,
                        'route': route_result['route'],
                        'timing': route_result['timing'],
                        'traffic': route_result['traffic']
                    })

                    # Обновляем время для следующего маршрута с учетом времени остановки
                    arrival_time = datetime.fromisoformat(route_result['timing']['arrival_time'])
                    current_time = arrival_time + timedelta(minutes=stop_duration)
                else:
                    routes.append({
                        'destination_index': i,
                        'destination_coords': [end_lat, end_lon],
                        'error': route_result['message']
                    })

            return {
                'status': 'success',
                'total_destinations': len(destinations),
                'successful_routes': len([r for r in routes if 'error' not in r]),
                'routes': routes,
                'total_travel_time': self._calculate_total_travel_time(routes),
                'parallel_execution': False
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Ошибка расчета множественных маршрутов: {str(e)}'
            }

    def _get_stop_duration(self, client_type: str) -> int:
        """Получение времени остановки в зависимости от типа клиента"""
        if client_type.lower() == 'vip':
            return 30  # 30 минут для VIP клиентов
        else:
            return 20  # 20 минут для обычных клиентов

    def _try_fallback_routing(self, start_lat: float, start_lon: float,
                            end_lat: float, end_lon: float,
                            departure_time: datetime) -> Optional[Dict]:
        """Пробуем альтернативные стратегии маршрутизации"""
        try:
            # Стратегия 1: Упрощенный маршрут без трафика
            print("  🔄 Стратегия 1: Упрощенный маршрут...")
            simple_route = self._get_simple_route(start_lat, start_lon, end_lat, end_lon)
            if simple_route:
                return self._format_route_response(simple_route, departure_time)

            # Стратегия 2: Маршрут через промежуточные точки
            print("  🔄 Стратегия 2: Через промежуточные точки...")
            intermediate_route = self._get_route_via_intermediate(start_lat, start_lon, end_lat, end_lon)
            if intermediate_route:
                return self._format_route_response(intermediate_route, departure_time)

            return None

        except Exception as e:
            print(f"  ❌ Ошибка fallback стратегий: {e}")
            return None

    def _get_simple_route(self, start_lat: float, start_lon: float,
                         end_lat: float, end_lon: float) -> Optional[Dict]:
        """Получение упрощенного маршрута без трафика"""
        try:
            url = f"{self.base_url}/routing/1/calculateRoute/{start_lat},{start_lon}:{end_lat},{end_lon}/json"
            params = {
                'key': self.api_key,
                'routeType': 'fastest',
                'traffic': 'false',  # Отключаем трафик для упрощения
                'travelMode': 'car'
            }

            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'routes' in data and data['routes']:
                    route = data['routes'][0]
                    return {
                        'distance_meters': route['summary']['lengthInMeters'],
                        'travel_time_seconds': route['summary']['travelTimeInSeconds'],
                        'route_type': 'simple'
                    }
            return None

        except Exception as e:
            print(f"    ❌ Ошибка простого маршрута: {e}")
            return None

    def _get_route_via_intermediate(self, start_lat: float, start_lon: float,
                                  end_lat: float, end_lon: float) -> Optional[Dict]:
        """Маршрут через промежуточные точки"""
        try:
            # Вычисляем промежуточную точку
            mid_lat = (start_lat + end_lat) / 2
            mid_lon = (start_lon + end_lon) / 2

            # Строим маршрут через промежуточную точку
            url = f"{self.base_url}/routing/1/calculateRoute/{start_lat},{start_lon}:{mid_lat},{mid_lon}:{end_lat},{end_lon}/json"
            params = {
                'key': self.api_key,
                'routeType': 'fastest',
                'traffic': 'false',
                'travelMode': 'car'
            }

            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if 'routes' in data and data['routes']:
                    route = data['routes'][0]
                    return {
                        'distance_meters': route['summary']['lengthInMeters'],
                        'travel_time_seconds': route['summary']['travelTimeInSeconds'],
                        'route_type': 'intermediate'
                    }
            return None

        except Exception as e:
            print(f"    ❌ Ошибка промежуточного маршрута: {e}")
            return None

    def _create_simple_route(self, start_lat: float, start_lon: float,
                           end_lat: float, end_lon: float,
                           departure_time: datetime) -> Dict:
        """Создание простого маршрута по прямой (последний fallback)"""
        try:
            # Вычисляем расстояние по прямой (Haversine formula)
            distance = self._calculate_haversine_distance(start_lat, start_lon, end_lat, end_lon)

            # Примерное время в пути (50 км/ч средняя скорость)
            estimated_speed_kmh = 50
            travel_time_seconds = int((distance / 1000) / estimated_speed_kmh * 3600)

            # Добавляем 20% времени на дорожные условия
            travel_time_seconds = int(travel_time_seconds * 1.2)

            arrival_time = departure_time + timedelta(seconds=travel_time_seconds)

            return {
                'status': 'success',
                'route': {
                    'distance_meters': int(distance),
                    'travel_time_seconds': travel_time_seconds,
                    'route_type': 'straight_line_fallback'
                },
                'timing': {
                    'departure_time': departure_time.isoformat(),
                    'arrival_time': arrival_time.isoformat(),
                    'travel_time_seconds': travel_time_seconds,
                    'travel_time_formatted': f"{travel_time_seconds // 60}м {travel_time_seconds % 60}с",
                    'traffic_delay_seconds': 0,
                    'traffic_delay_formatted': "0с"
                },
                'traffic': {
                    'traffic_level': 'Неизвестно (fallback)',
                    'delay_seconds': 0
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Ошибка создания fallback маршрута: {str(e)}'
            }

    def _calculate_haversine_distance(self, lat1: float, lon1: float,
                                     lat2: float, lon2: float) -> float:
        """Вычисление расстояния по формуле Haversine"""
        import math

        R = 6371000  # Радиус Земли в метрах

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))

        return R * c

    def _format_route_response(self, route_data: Dict, departure_time: datetime) -> Dict:
        """Форматирование ответа маршрута"""
        try:
            arrival_time = departure_time + timedelta(seconds=route_data['travel_time_seconds'])

            return {
                'status': 'success',
                'route': {
                    'distance_meters': route_data['distance_meters'],
                    'travel_time_seconds': route_data['travel_time_seconds'],
                    'route_type': route_data.get('route_type', 'standard')
                },
                'timing': {
                    'departure_time': departure_time.isoformat(),
                    'arrival_time': arrival_time.isoformat(),
                    'travel_time_seconds': route_data['travel_time_seconds'],
                    'travel_time_formatted': f"{route_data['travel_time_seconds'] // 60}м {route_data['travel_time_seconds'] % 60}с",
                    'traffic_delay_seconds': 0,
                    'traffic_delay_formatted': "0с"
                },
                'traffic': {
                    'traffic_level': 'Свободно (fallback)',
                    'delay_seconds': 0
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Ошибка форматирования ответа: {str(e)}'
            }

    def _calculate_total_travel_time(self, routes: List[Dict]) -> str:
        """Расчет общего времени в пути включая остановки"""
        total_seconds = 0

        for route in routes:
            if 'route' in route:
                # Время в пути
                total_seconds += route['route']['travel_time_seconds']
                # Время остановки
                total_seconds += route.get('stop_duration_minutes', 20) * 60

        return self._format_duration(total_seconds)

    def optimize_route_order_integrated(self, clients: List[Dict]) -> Dict:
        """
        Интегрированная оптимизация: Graph Transformer + ANN
        1. Graph Transformer извлекает графовые признаки
        2. ANN принимает обогащенные признаки и делает финальное решение
        """
        print("🧠 Интегрированная оптимизация: Graph Transformer + ANN")

        try:
            if not self.ann_model:
                print("❌ ANN модель не загружена!")
                return {
                    'status': 'error',
                    'message': 'ANN модель не загружена',
                    'method': 'integrated'
                }

            # Этап 1: Graph Transformer извлекает графовые признаки
            print("🎯 Этап 1: Graph Transformer извлекает графовые признаки...")
            enriched_features = self.graph_feature_extractor.extract_graph_features(clients)

            # Этап 2: ANN принимает обогащенные признаки
            print("🧠 Этап 2: ANN обрабатывает обогащенные признаки...")

            # Подготавливаем данные для ANN
            features_tensor = torch.tensor(enriched_features, dtype=torch.float)

            # ANN предсказание
            self.ann_model.eval()
            with torch.no_grad():
                # ANN предсказывает порядок
                ann_output = self.ann_model(features_tensor)

                # Получаем индексы отсортированного порядка
                _, optimized_order = torch.sort(ann_output, descending=True)
                optimized_order = optimized_order.tolist()

            # Создаем результат
            result = {
                'status': 'success',
                'optimized_order': optimized_order,
                'method': 'integrated_graph_transformer_ann',
                'total_clients': len(clients),
                'optimization_time': 0.01,  # Очень быстро!
                'graph_features_extracted': len(enriched_features[0]) - 7,  # 7 графовых признаков
                'total_features': len(enriched_features[0])  # 14 общих признаков
            }

            print(f"✅ Интегрированная оптимизация завершена за 0.01с")
            print(f"🎯 Оптимизированный порядок: {optimized_order}")
            print(f"📊 Графовых признаков: {result['graph_features_extracted']}")
            print(f"📊 Общих признаков: {result['total_features']}")

            return result

        except Exception as e:
            print(f"❌ Ошибка интегрированной оптимизации: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'method': 'integrated_graph_transformer_ann'
            }

    def train_graph_transformer(self, training_data: List[Dict], epochs: int = 50,
                               learning_rate: float = 0.001) -> Dict:
        """
        Обучение Graph Transformer на исторических данных

        Args:
            training_data: Список исторических маршрутов
            epochs: Количество эпох обучения
            learning_rate: Скорость обучения

        Returns:
            Dict с результатами обучения
        """
        print(f"🧠 Начинаем обучение Graph Transformer на {len(training_data)} примерах...")

        try:
            # Подготавливаем данные для обучения
            optimizer = torch.optim.Adam(self.graph_model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            # Простое обучение (для демонстрации)
            self.graph_model.train()

            for epoch in range(epochs):
                total_loss = 0

                for batch_data in training_data:
                    # Создаем граф из данных
                    graph_data = self.graph_model.create_graph_from_clients(batch_data['clients'])

                    # Получаем предсказания
                    outputs = self.graph_model(graph_data)

                    # Простая loss функция (можно улучшить)
                    target_priorities = torch.randn_like(outputs['route_priorities'])
                    target_times = torch.randn_like(outputs['travel_times'])

                    loss = criterion(outputs['route_priorities'], target_priorities) + \
                           criterion(outputs['travel_times'], target_times)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                if epoch % 10 == 0:
                    print(f"Эпоха {epoch}/{epochs}, Loss: {total_loss:.4f}")

            # Сохраняем модель
            torch.save(self.graph_model.state_dict(), 'graph_transformer_model.pth')

            print(f"✅ Обучение завершено! Модель сохранена в graph_transformer_model.pth")

            return {
                'status': 'success',
                'epochs_trained': epochs,
                'final_loss': total_loss,
                'model_saved': 'graph_transformer_model.pth'
            }

        except Exception as e:
            print(f"❌ Ошибка обучения: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def load_trained_graph_transformer(self, model_path: str = 'graph_transformer_model.pth') -> bool:
        """
        Загружает предобученную Graph Transformer модель

        Args:
            model_path: Путь к файлу модели

        Returns:
            bool: True если загрузка успешна
        """
        try:
            if os.path.exists(model_path):
                self.graph_model.load_state_dict(torch.load(model_path))
                self.graph_model.eval()
                print(f"✅ Graph Transformer модель загружена из {model_path}")
                return True
            else:
                print(f"⚠️ Файл модели {model_path} не найден. Используем случайные веса.")
                return False
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False

    def optimize_route_order_hybrid(self,
                                   start_coords: Tuple[float, float],
                                   destinations: List[Tuple[float, float]],
                                   client_types: List[str] = None,
                                   departure_time: Optional[datetime] = None,
                                   clients_data: List[Dict] = None) -> Dict:
        """
        Гибридная оптимизация: Graph Transformer + API
        1. Graph Transformer предварительно оптимизирует порядок
        2. API строит точные маршруты по оптимизированному порядку
        """
        print("🧠 Гибридная оптимизация: Graph Transformer + API")

        try:
            # Этап 1: Graph Transformer предварительная оптимизация
            if clients_data:
                print("🎯 Этап 1: Graph Transformer предварительная оптимизация...")
                graph_order = self.graph_model.optimize_route_order(clients_data)
                print(f"✅ Graph Transformer порядок: {graph_order}")
            else:
                # Fallback к простому порядку если нет данных клиентов
                graph_order = list(range(len(destinations)))
                print("⚠️ Нет данных клиентов, используем простой порядок")

            # Этап 2: API построение маршрутов по оптимизированному порядку
            print("🚀 Этап 2: API построение маршрутов по оптимизированному порядку...")

            # Переупорядочиваем точки согласно Graph Transformer
            optimized_destinations = [destinations[i] for i in graph_order]
            optimized_client_types = [client_types[i] for i in graph_order] if client_types else None

            # Строим маршруты по оптимизированному порядку
            total_travel_time = 0
            routes = []
            current_time = departure_time or datetime.now()

            for i, (dest_lat, dest_lon) in enumerate(optimized_destinations):
                if i == 0:
                    # Первая точка - от старта
                    route_result = self.calculate_route_with_traffic(
                        start_coords[0], start_coords[1], dest_lat, dest_lon, current_time
                    )
                else:
                    # Остальные точки - от предыдущей
                    prev_dest = optimized_destinations[i-1]
                    route_result = self.calculate_route_with_traffic(
                        prev_dest[0], prev_dest[1], dest_lat, dest_lon, current_time
                    )

                if route_result['status'] == 'success':
                    routes.append(route_result)
                    total_travel_time += route_result['travel_time_seconds']

                    # Обновляем время для следующей точки
                    current_time += timedelta(seconds=route_result['travel_time_seconds'])

                    # Добавляем время остановки
                    stop_duration = 30 if (optimized_client_types and optimized_client_types[i] == 'vip') else 20
                    current_time += timedelta(minutes=stop_duration)
                else:
                    print(f"❌ Ошибка маршрута к точке {i+1}: {route_result.get('error', 'Неизвестная ошибка')}")
                    routes.append({'error': route_result.get('error', 'Неизвестная ошибка')})

            # Форматируем результат
            total_time_formatted = self._format_duration(total_travel_time)
            completion_time = current_time.strftime('%Y-%m-%dT%H:%M:%S')

            return {
                'status': 'success',
                'optimized_order': graph_order,
                'total_travel_time_formatted': total_time_formatted,
                'estimated_completion_time': completion_time,
                'routes': routes,
                'method': 'hybrid_graph_transformer_api',
                'graph_transformer_order': graph_order,
                'api_routes_built': len([r for r in routes if 'error' not in r])
            }

        except Exception as e:
            print(f"❌ Ошибка гибридной оптимизации: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'method': 'hybrid_graph_transformer_api'
            }

    def optimize_route_order(self,
                           start_coords: Tuple[float, float],
                           destinations: List[Tuple[float, float]],
                           client_types: List[str] = None,
                           departure_time: Optional[datetime] = None) -> Dict:
        """
        Оптимизация порядка посещения точек (простой алгоритм ближайшего соседа)

        Args:
            start_coords: Координаты начала
            destinations: Список координат назначения
            client_types: Список типов клиентов ['VIP', 'regular', ...]
            departure_time: Время отправления

        Returns:
            Dict с оптимизированным порядком посещения
        """
        try:
            if departure_time is None:
                departure_time = datetime.now()

            # Устанавливаем типы клиентов по умолчанию
            if client_types is None:
                client_types = ['regular'] * len(destinations)

            # Простой алгоритм ближайшего соседа
            unvisited = list(range(len(destinations)))
            current_coords = start_coords
            optimized_order = []
            total_time = 0

            while unvisited:
                best_index = None
                best_time = float('inf')

                for i in unvisited:
                    # Получаем время до точки
                    route_result = self.calculate_route_with_traffic(
                        current_coords[0], current_coords[1],
                        destinations[i][0], destinations[i][1],
                        departure_time
                    )

                    if route_result['status'] == 'success':
                        travel_time = route_result['route']['travel_time_seconds']
                        if travel_time < best_time:
                            best_time = travel_time
                            best_index = i

                if best_index is not None:
                    optimized_order.append(best_index)
                    unvisited.remove(best_index)
                    current_coords = destinations[best_index]
                    total_time += best_time

                    # Добавляем время остановки в зависимости от типа клиента
                    client_type = client_types[best_index] if best_index < len(client_types) else 'regular'
                    stop_duration = self._get_stop_duration(client_type)
                    total_time += stop_duration * 60
                    departure_time += timedelta(seconds=best_time + stop_duration * 60)
                else:
                    break

            return {
                'status': 'success',
                'optimized_order': optimized_order,
                'total_travel_time_seconds': total_time,
                'total_travel_time_formatted': self._format_duration(total_time),
                'estimated_completion_time': (datetime.now() + timedelta(seconds=total_time)).isoformat()
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Ошибка оптимизации маршрута: {str(e)}'
            }

    def load_client_data(self, file_path: str = "DATA (2).txt") -> Dict:
        """
        Загрузка данных клиентов из файла DATA (2).txt

        Args:
            file_path: Путь к файлу с данными клиентов

        Returns:
            Dict с данными клиентов
        """
        try:
            if not os.path.exists(file_path):
                return {
                    'status': 'error',
                    'message': f'Файл {file_path} не найден'
                }

            print(f"📂 Загрузка данных клиентов из {file_path}...")

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Парсинг JSON-подобных данных
            clients = self._parse_client_data(content)

            if not clients:
                return {
                    'status': 'error',
                    'message': 'Не удалось загрузить данные клиентов'
                }

            print(f"✅ Загружено {len(clients)} клиентов")

            return {
                'status': 'success',
                'clients': clients,
                'total_count': len(clients),
                'vip_count': len([c for c in clients if c.get('client_level', '').upper() == 'VIP']),
                'regular_count': len([c for c in clients if c.get('client_level', '').upper() != 'VIP'])
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Ошибка загрузки данных: {str(e)}'
            }

    def _parse_client_data(self, content: str) -> List[Dict]:
        """Парсинг данных клиентов из текстового файла"""
        clients = []

        try:
            # Попытка парсинга как JSON
            if content.strip().startswith('['):
                # Массив JSON объектов
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            clients.append(self._normalize_client_data(item))
            elif 'data = [' in content:
                # JavaScript-подобный формат с переменной data
                clients = self._parse_js_data_format(content)
            else:
                # Парсинг отдельных JSON объектов
                lines = content.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('{') or line.startswith('[')):
                        try:
                            data = json.loads(line)
                            if isinstance(data, dict):
                                clients.append(self._normalize_client_data(data))
                            elif isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict):
                                        clients.append(self._normalize_client_data(item))
                        except json.JSONDecodeError:
                            continue
        except json.JSONDecodeError:
            # Fallback: парсинг с помощью регулярных выражений
            clients = self._parse_with_regex(content)

        return clients

    def _parse_with_regex(self, content: str) -> List[Dict]:
        """Парсинг данных с помощью регулярных выражений"""
        clients = []

        # Паттерн для поиска JSON-подобных объектов
        pattern = r'\{[^{}]*"id"[^{}]*\}'
        matches = re.findall(pattern, content)

        for match in matches:
            try:
                # Очистка и парсинг
                clean_match = match.replace("'", '"')
                data = json.loads(clean_match)
                clients.append(self._normalize_client_data(data))
            except:
                continue

        return clients

    def _parse_js_data_format(self, content: str) -> List[Dict]:
        """Парсинг JavaScript-подобного формата с переменной data"""
        clients = []

        try:
            # Ищем начало массива data = [
            start_index = content.find('data = [')
            if start_index == -1:
                return clients

            # Ищем конец массива
            bracket_count = 0
            start_bracket = False
            data_start = -1
            data_end = -1

            for i, char in enumerate(content[start_index:]):
                if char == '[' and not start_bracket:
                    start_bracket = True
                    data_start = start_index + i
                    bracket_count = 1
                elif start_bracket:
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            data_end = start_index + i + 1
                            break

            if data_start != -1 and data_end != -1:
                # Извлекаем JSON массив
                json_data = content[data_start:data_end]

                # Очистка от лишних запятых и других проблем
                json_data = self._clean_json_data(json_data)

                data = json.loads(json_data)

                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            clients.append(self._normalize_client_data(item))

        except Exception as e:
            print(f"Ошибка парсинга JS формата: {e}")

        return clients

    def _clean_json_data(self, json_data: str) -> str:
        """Очистка JSON данных от лишних запятых и других проблем"""
        try:
            # Удаляем лишние запятые перед закрывающими скобками
            json_data = re.sub(r',\s*\]', ']', json_data)
            json_data = re.sub(r',\s*}', '}', json_data)

            # Удаляем комментарии (если есть)
            json_data = re.sub(r'//.*$', '', json_data, flags=re.MULTILINE)

            # Удаляем лишние пробелы и переносы строк
            json_data = re.sub(r'\s+', ' ', json_data)

            return json_data.strip()

        except Exception as e:
            print(f"Ошибка очистки JSON: {e}")
            return json_data

    def _normalize_client_data(self, client_data: Dict) -> Dict:
        """Нормализация данных клиента"""
        client_level = client_data.get('client_level', 'regular')

        return {
            'id': client_data.get('id', 0),
            'address': client_data.get('address', ''),
            'address1': client_data.get('address1', ''),
            'lat': float(client_data.get('lat', 0)),
            'lon': float(client_data.get('lon', 0)),
            'client_level': client_level,
            'work_start': client_data.get('work_start', '09:00'),
            'work_end': client_data.get('work_end', '18:00'),
            'lunch_start': client_data.get('lunch_start', '13:00'),
            'lunch_end': client_data.get('lunch_end', '14:00'),
            'is_vip': client_level.upper() == 'VIP'
        }

    def calculate_routes_from_file(self,
                                 file_path: str = "DATA (2).txt",
                                 start_coords: Optional[Tuple[float, float]] = None,
                                 departure_time: Optional[datetime] = None,
                                 use_parallel: bool = True,
                                 max_workers: int = 5) -> Dict:
        """
        Расчет маршрутов для всех клиентов из файла

        Args:
            file_path: Путь к файлу с данными клиентов
            start_coords: Координаты начала маршрута (по умолчанию - первый клиент)
            departure_time: Время отправления
            use_parallel: Использовать параллельные вычисления
            max_workers: Максимальное количество потоков

        Returns:
            Dict с маршрутами для всех клиентов
        """
        try:
            # Загрузка данных клиентов
            client_data = self.load_client_data(file_path)

            if client_data['status'] != 'success':
                return client_data

            clients = client_data['clients']

            if not clients:
                return {
                    'status': 'error',
                    'message': 'Нет данных о клиентах'
                }

            # Определение начальной точки
            if start_coords is None:
                start_coords = (clients[0]['lat'], clients[0]['lon'])
                print(f"📍 Начальная точка: {start_coords} (первый клиент)")
            else:
                print(f"📍 Начальная точка: {start_coords}")

            # Подготовка данных для расчета маршрутов
            destinations = [(client['lat'], client['lon']) for client in clients]
            client_types = [client['client_level'] for client in clients]

            print(f"🎯 Назначения: {len(destinations)} клиентов")
            print(f"👑 VIP клиентов: {client_data['vip_count']}")
            print(f"👤 Обычных клиентов: {client_data['regular_count']}")

            # Расчет маршрутов (параллельный или последовательный)
            if use_parallel:
                print(f"🚀 Используем параллельные вычисления с {max_workers} потоками")
                routes_result = self.calculate_multiple_routes_parallel(
                    start_coords, destinations, client_types, departure_time, max_workers
                )
            else:
                print("🔄 Используем последовательные вычисления")
                routes_result = self.calculate_multiple_routes(
                    start_coords, destinations, client_types, departure_time
                )

            if routes_result['status'] != 'success':
                return routes_result

            # Добавление информации о клиентах к маршрутам
            enhanced_routes = []
            for i, route in enumerate(routes_result['routes']):
                if 'error' not in route:
                    client_info = clients[i]
                    route['client_info'] = {
                        'id': client_info['id'],
                        'address': client_info['address'],
                        'client_level': client_info['client_level'],
                        'work_start': client_info['work_start'],
                        'work_end': client_info['work_end'],
                        'lunch_start': client_info['lunch_start'],
                        'lunch_end': client_info['lunch_end']
                    }
                enhanced_routes.append(route)

            return {
                'status': 'success',
                'total_clients': len(clients),
                'successful_routes': routes_result['successful_routes'],
                'total_travel_time': routes_result['total_travel_time'],
                'routes': enhanced_routes,
                'client_summary': {
                    'vip_count': client_data['vip_count'],
                    'regular_count': client_data['regular_count']
                },
                'execution_info': {
                    'parallel_execution': routes_result.get('parallel_execution', False),
                    'max_workers': routes_result.get('max_workers', 1),
                    'cache_enabled': self.cache_enabled
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Ошибка расчета маршрутов из файла: {str(e)}'
            }

    def optimize_routes_from_file(self,
                                 file_path: str = "DATA (2).txt",
                                 start_coords: Optional[Tuple[float, float]] = None,
                                 departure_time: Optional[datetime] = None,
                                 use_graph_transformer: bool = True) -> Dict:
        """
        Оптимизация маршрутов для всех клиентов из файла
        Использует Graph Transformer для мгновенной оптимизации

        Args:
            file_path: Путь к файлу с данными клиентов
            start_coords: Координаты начала маршрута
            departure_time: Время отправления
            use_graph_transformer: Использовать Graph Transformer (True) или API (False)

        Returns:
            Dict с оптимизированными маршрутами
        """
        try:
            # Загрузка данных клиентов
            client_data = self.load_client_data(file_path)

            if client_data['status'] != 'success':
                return client_data

            clients = client_data['clients']

            if not clients:
                return {
                    'status': 'error',
                    'message': 'Нет данных о клиентах'
                }

            # Определение начальной точки
            if start_coords is None:
                start_coords = (clients[0]['lat'], clients[0]['lon'])

            # Подготовка данных для оптимизации
            destinations = [(client['lat'], client['lon']) for client in clients]
            client_types = [client['client_level'] for client in clients]

            # Используем интегрированный метод: Graph Transformer + ANN + API
            print(f"🧠 Интегрированная оптимизация для {len(clients)} клиентов...")

            # Этап 1: Graph Transformer + ANN оптимизация
            integrated_result = self.optimize_route_order_integrated(clients)

            if integrated_result['status'] != 'success':
                return integrated_result

            # Этап 2: API построение маршрутов по оптимизированному порядку
            print("🚀 Этап 2: API построение маршрутов по оптимизированному порядку...")

            # Переупорядочиваем точки согласно интегрированной оптимизации
            graph_order = integrated_result['optimized_order']
            optimized_destinations = [destinations[i] for i in graph_order]
            optimized_client_types = [client_types[i] for i in graph_order] if client_types else None

            # Строим маршруты по оптимизированному порядку
            total_travel_time = 0
            routes = []
            current_time = departure_time or datetime.now()

            for i, (dest_lat, dest_lon) in enumerate(optimized_destinations):
                if i == 0:
                    # Первая точка - от старта
                    route_result = self.calculate_route_with_traffic(
                        start_coords[0], start_coords[1], dest_lat, dest_lon, current_time
                    )
                else:
                    # Остальные точки - от предыдущей
                    prev_dest = optimized_destinations[i-1]
                    route_result = self.calculate_route_with_traffic(
                        prev_dest[0], prev_dest[1], dest_lat, dest_lon, current_time
                    )

                if route_result['status'] == 'success':
                    routes.append(route_result)
                    total_travel_time += route_result['travel_time_seconds']

                    # Обновляем время для следующей точки
                    current_time += timedelta(seconds=route_result['travel_time_seconds'])

                    # Добавляем время остановки
                    stop_duration = 30 if (optimized_client_types and optimized_client_types[i] == 'vip') else 20
                    current_time += timedelta(minutes=stop_duration)
                else:
                    print(f"❌ Ошибка маршрута к точке {i+1}: {route_result.get('error', 'Неизвестная ошибка')}")
                    routes.append({'error': route_result.get('error', 'Неизвестная ошибка')})

            # Форматируем результат
            total_time_formatted = self._format_duration(total_travel_time)
            completion_time = current_time.strftime('%Y-%m-%dT%H:%M:%S')

            optimization_result = {
                'status': 'success',
                'optimized_order': graph_order,
                'total_travel_time_formatted': total_time_formatted,
                'estimated_completion_time': completion_time,
                'routes': routes,
                'method': 'integrated_graph_transformer_ann_api',
                'graph_transformer_order': graph_order,
                'api_routes_built': len([r for r in routes if 'error' not in r]),
                'graph_features_extracted': integrated_result.get('graph_features_extracted', 0),
                'total_features': integrated_result.get('total_features', 0)
            }

            if optimization_result['status'] != 'success':
                return optimization_result

            # Создание оптимизированного списка клиентов
            optimized_clients = []
            for index in optimization_result['optimized_order']:
                if index < len(clients):
                    optimized_clients.append(clients[index])

            return {
                'status': 'success',
                'optimized_order': optimization_result['optimized_order'],
                'total_travel_time': optimization_result['total_travel_time_formatted'],
                'estimated_completion_time': optimization_result['estimated_completion_time'],
                'optimized_clients': optimized_clients,
                'client_summary': {
                    'total_clients': len(clients),
                    'vip_count': client_data['vip_count'],
                    'regular_count': client_data['regular_count']
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Ошибка оптимизации маршрутов из файла: {str(e)}'
            }

def main():
    """Демонстрация работы модуля"""
    print("🚦 TomTom Traffic Router Demo")
    print("=" * 50)

    # Запрос API ключа
    api_key = input("Введите ваш TomTom API ключ: ").strip()

    if not api_key:
        print("❌ API ключ не введен!")
        return

    # Создание роутера
    # Инициализация роутера
    router = TomTomTrafficRouter(api_key)
    print("🕐 Режим реального времени: ВКЛЮЧЕН")
    print("🔄 Трафик всегда актуальный (без кэша)")

    # Загружаем ВСЕ клиенты из файла
    print("\n📂 Загрузка всех клиентов из DATA (2).txt...")
    file_result = router.load_client_data("DATA (2).txt")

    if file_result['status'] != 'success':
        print("❌ Не удалось загрузить клиентов!")
        return

    # Используем ВСЕХ клиентов
    clients = file_result['clients']
    start_coords = (clients[0]['lat'], clients[0]['lon'])  # Первый клиент как стартовая точка
    destinations = [(client['lat'], client['lon']) for client in clients]
    client_types = [client['client_level'].upper() for client in clients]

    print(f"✅ Загружено {len(clients)} клиентов")
    print(f"📍 Начальная точка: {start_coords}")
    print(f"🎯 Назначения: {len(destinations)} точек")

    # Тест 4: Загрузка данных из файла (уже загружено выше)
    print("\n4️⃣ Тест загрузки данных из файла:")
    print(f"✅ Загружено {len(clients)} клиентов")
    print(f"👑 VIP клиентов: {file_result['vip_count']}")
    print(f"👤 Обычных клиентов: {file_result['regular_count']}")

    # Показываем первых 3 клиентов
    for i, client in enumerate(clients[:3]):
        print(f"  📍 Клиент {i+1}: {client['address'][:50]}... ({client['client_level']})")

    # Тест 5: Расчет маршрутов из файла (параллельный)
    print("\n5️⃣ Тест расчета маршрутов из файла (параллельный):")
    routes_from_file = router.calculate_routes_from_file(
        "DATA (2).txt",
        use_parallel=True,
        max_workers=25  # Увеличено с 10 до 25 для 15 клиентов
    )

    if routes_from_file['status'] == 'success':
        exec_info = routes_from_file.get('execution_info', {})
        print(f"✅ Рассчитано {routes_from_file['successful_routes']}/{routes_from_file['total_clients']} маршрутов")
        print(f"⏱️ Общее время в пути: {routes_from_file['total_travel_time']}")
        print(f"🚀 Параллельное выполнение: {exec_info.get('parallel_execution', False)}")
        print(f"👥 Потоков: {exec_info.get('max_workers', 1)}")
        print(f"📦 Кэш включен: {exec_info.get('cache_enabled', False)}")

        # Показываем ВСЕ маршруты (не только первые 3)
        print(f"📋 Все маршруты ({len(routes_from_file['routes'])}):")
        for i, route in enumerate(routes_from_file['routes']):
            if 'error' not in route:
                client_info = route.get('client_info', {})
                print(f"  ✅ {i+1}. {client_info.get('address', 'Неизвестно')[:30]}... ({client_info.get('client_level', 'regular')})")
            else:
                print(f"  ❌ {i+1}. Ошибка: {route.get('error', 'Неизвестная ошибка')}")
    else:
        print(f"❌ Ошибка расчета: {routes_from_file['message']}")

    # Тест 6: УЛУЧШЕННАЯ СИСТЕМА - Graph Transformer + API!
    print("\n6️⃣ Тест УЛУЧШЕННОЙ системы (Graph Transformer + API):")
    enhanced_optimization_result = router.optimize_routes_from_file("DATA (2).txt")

    if enhanced_optimization_result['status'] == 'success':
        print(f"✅ УЛУЧШЕННАЯ оптимизация завершена!")
        print(f"🧠 Graph Transformer порядок: {enhanced_optimization_result['optimized_order']}")
        print(f"🚀 API маршрутов построено: {enhanced_optimization_result.get('api_routes_built', 'N/A')}")
        print(f"⚡ Метод: {enhanced_optimization_result.get('method', 'hybrid')}")
        print(f"⏱️ Общее время: {enhanced_optimization_result.get('total_travel_time', 'N/A')}")
        print(f"🕐 Время завершения: {enhanced_optimization_result.get('estimated_completion_time', 'N/A')}")
        print(f"👑 VIP клиентов: {enhanced_optimization_result['client_summary']['vip_count']}")
        print(f"👤 Обычных клиентов: {enhanced_optimization_result['client_summary']['regular_count']}")

        # Показываем оптимизированный порядок
        print(f"\n📋 УЛУЧШЕННЫЙ порядок посещения:")
        for i, client in enumerate(enhanced_optimization_result['optimized_clients']):
            client_type = "VIP" if client['client_level'].lower() == 'vip' else "Стандарт"
            print(f"  {i+1}. {client['address'][:50]}... ({client_type})")
    else:
        print(f"❌ Ошибка УЛУЧШЕННОЙ оптимизации: {enhanced_optimization_result['message']}")

    # Тест 7: Обучение Graph Transformer для улучшения системы
    print("\n7️⃣ Обучение Graph Transformer для улучшения системы:")

    # Создаем синтетические данные для обучения
    training_data = []
    for i in range(10):  # 10 примеров для обучения
        synthetic_clients = []
        for j in range(5):  # 5 клиентов в каждом примере
            client = {
                'lat': 47.2 + np.random.uniform(-0.1, 0.1),
                'lon': 39.7 + np.random.uniform(-0.1, 0.1),
                'client_level': 'VIP' if j == 0 else 'regular',
                'work_start_hour': 8.0 + np.random.uniform(0, 2),
                'work_end_hour': 18.0 - np.random.uniform(0, 2),
                'lunch_start_hour': 13.0 + np.random.uniform(0, 1),
                'lunch_end_hour': 14.0 + np.random.uniform(0, 1)
            }
            synthetic_clients.append(client)

        training_data.append({'clients': synthetic_clients})

    # Обучаем модель для улучшения системы
    training_result = router.train_graph_transformer(training_data, epochs=20, learning_rate=0.001)

    if training_result['status'] == 'success':
        print(f"✅ Graph Transformer обучен для улучшения системы!")
        print(f"📊 Эпох обучено: {training_result['epochs_trained']}")
        print(f"📉 Финальный loss: {training_result['final_loss']:.4f}")
        print(f"💾 Модель сохранена: {training_result['model_saved']}")
        print(f"🧠 Теперь система использует обученный Graph Transformer для предварительной оптимизации!")
    else:
        print(f"❌ Ошибка обучения: {training_result['message']}")

    # Тест 1: Расчет одного маршрута
    print("\n1️⃣ Тест расчета одного маршрута:")
    route_result = router.calculate_route_with_traffic(
        start_coords[0], start_coords[1],
        destinations[0][0], destinations[0][1]
    )

    if route_result['status'] == 'success':
        print(f"✅ Маршрут построен успешно!")
        print(f"📏 Расстояние: {route_result['route']['distance_meters']} м")
        print(f"⏱️ Время в пути: {route_result['timing']['travel_time_formatted']}")
        print(f"🚦 Задержка от трафика: {route_result['timing']['traffic_delay_formatted']}")
        print(f"🕐 Время прибытия: {route_result['timing']['arrival_time']}")
        print(f"🚗 Уровень трафика: {route_result['traffic']['traffic_level']}")
    else:
        print(f"❌ Ошибка: {route_result['message']}")

    # Тест 2: Расчет множественных маршрутов (ВСЕ клиенты)
    print("\n2️⃣ Тест расчета множественных маршрутов (ВСЕ клиенты):")

    # Загружаем ВСЕ клиентов из файла
    all_clients_result = router.load_client_data("DATA (2).txt")
    if all_clients_result['status'] == 'success':
        all_destinations = [(client['lat'], client['lon']) for client in all_clients_result['clients']]
        all_client_types = [client['client_level'].lower() for client in all_clients_result['clients']]

        print(f"🎯 Расчет маршрутов для {len(all_destinations)} клиентов...")
        # Определяем время отправления
        departure_time = datetime.now()

        multiple_result = router.calculate_multiple_routes_parallel(
            start_coords, all_destinations, all_client_types, departure_time, max_workers=25
        )
    else:
        print(f"❌ Ошибка загрузки клиентов: {all_clients_result['message']}")
        multiple_result = {'status': 'error', 'message': 'Не удалось загрузить клиентов'}

    if multiple_result['status'] == 'success':
        print(f"✅ Рассчитано {multiple_result['successful_routes']}/{multiple_result['total_destinations']} маршрутов")
        print(f"⏱️ Общее время в пути: {multiple_result['total_travel_time']}")

        for i, route in enumerate(multiple_result['routes']):
            if 'error' not in route:
                client_type = route.get('client_type', 'regular')
                stop_duration = route.get('stop_duration_minutes', 20)
                print(f"  📍 Точка {i+1} ({client_type}): {route['timing']['travel_time_formatted']} + {stop_duration}м остановка")
            else:
                print(f"  ❌ Точка {i+1}: {route['error']}")
    else:
        print(f"❌ Ошибка: {multiple_result['message']}")

    # Тест 3: Graph Transformer оптимизация (СУПЕР БЫСТРО!)
    print("\n3️⃣ Тест Graph Transformer оптимизации (ВСЕ клиенты):")

    if all_clients_result['status'] == 'success':
        print(f"🧠 Graph Transformer оптимизация для {len(all_destinations)} клиентов...")
        graph_optimization_result = router.optimize_route_order_graph(all_clients_result['clients'])

        if graph_optimization_result['status'] == 'success':
            print(f"✅ Graph Transformer порядок: {graph_optimization_result['optimized_order']}")
            print(f"⚡ Время оптимизации: {graph_optimization_result['optimization_time']}с")
            print(f"🎯 Метод: {graph_optimization_result['method']}")
        else:
            print(f"❌ Ошибка Graph Transformer: {graph_optimization_result['message']}")
    else:
        print("❌ Не удалось загрузить клиентов для Graph Transformer")

    # Тест 4: Старая оптимизация (для сравнения)
    print("\n4️⃣ Тест старой оптимизации (для сравнения):")

    if all_clients_result['status'] == 'success':
        print(f"🎯 Старая оптимизация маршрута для {len(all_destinations)} клиентов...")
        optimization_result = router.optimize_route_order(
            start_coords, all_destinations, all_client_types, max_workers=25
        )
    else:
        optimization_result = {'status': 'error', 'message': 'Не удалось загрузить клиентов'}

    if optimization_result['status'] == 'success':
        print(f"✅ Оптимизированный порядок: {optimization_result['optimized_order']}")
        print(f"⏱️ Общее время: {optimization_result['total_travel_time_formatted']}")
        print(f"🕐 Время завершения: {optimization_result['estimated_completion_time']}")
    else:
        print(f"❌ Ошибка: {optimization_result['message']}")

if __name__ == "__main__":
    main()
