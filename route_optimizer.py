#!/usr/bin/env python3
"""
🚀 Система оптимизации маршрутов с распределением по дням
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import json
import time
from dataclasses import dataclass
from enum import Enum

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
    work_start: str  # "09:00"
    work_end: str    # "18:00"
    lunch_start: str # "13:00"
    lunch_end: str   # "14:00"

    @property
    def service_time_minutes(self) -> int:
        """Время обслуживания в минутах"""
        return 30 if self.client_level == ClientLevel.VIP else 20

    @property
    def work_start_hour(self) -> float:
        """Начало работы в часах (9.0)"""
        hour, minute = map(int, self.work_start.split(':'))
        return hour + minute / 60.0

    @property
    def work_end_hour(self) -> float:
        """Конец работы в часах (18.0)"""
        hour, minute = map(int, self.work_end.split(':'))
        return hour + minute / 60.0

    @property
    def lunch_start_hour(self) -> float:
        """Начало обеда в часах (13.0)"""
        hour, minute = map(int, self.lunch_start.split(':'))
        return hour + minute / 60.0

    @property
    def lunch_end_hour(self) -> float:
        """Конец обеда в часах (14.0)"""
        hour, minute = map(int, self.lunch_end.split(':'))
        return hour + minute / 60.0

@dataclass
class Route:
    """Маршрут на день"""
    day: int
    clients: List[Client]
    total_distance: float
    total_time: float
    estimated_completion: str

class RouteOptimizer:
    """Оптимизатор маршрутов с распределением по дням"""

    def __init__(self, tomtom_api_key: str = None):
        self.tomtom_api_key = tomtom_api_key
        self.clients = []
        self.current_routes = {}  # Текущие маршруты
        self.visited_clients = set()  # Посещенные клиенты
        self.current_time = 9.0  # Текущее время (9:00)

    def load_clients_from_file(self, file_path: str) -> List[Client]:
        """Загружает клиентов из DATA (2).txt"""
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

                data = json.loads(json_str)

                for item in data:
                    client = Client(
                        id=int(item['id']),
                        address=item['address'],
                        lat=float(item['lat']),
                        lon=float(item['lon']),
                        client_level=ClientLevel.VIP if item['client_level'] == 'VIP' else ClientLevel.REGULAR,
                        work_start=item['work_start'],
                        work_end=item['work_end'],
                        lunch_start=item['lunch_start'],
                        lunch_end=item['lunch_end']
                    )
                    clients.append(client)

            print(f"✅ Загружено {len(clients)} клиентов из {file_path}")
            return clients

        except Exception as e:
            print(f"❌ Ошибка загрузки клиентов: {e}")
            return []

    def distribute_clients_by_days(self, clients: List[Client], num_days: int) -> List[List[Client]]:
        """
        Распределяет клиентов по дням оптимально
        20 клиентов на 3 дня = 7+7+6 (не 4+4+4+4+4)
        """
        print(f"📅 Распределение {len(clients)} клиентов на {num_days} дней...")

        # Сортируем клиентов по приоритету (VIP сначала, потом по координатам)
        sorted_clients = sorted(clients, key=lambda c: (
            0 if c.client_level == ClientLevel.VIP else 1,  # VIP сначала
            c.lat, c.lon  # Потом по координатам
        ))

        # Вычисляем оптимальное распределение
        base_per_day = len(clients) // num_days
        extra_clients = len(clients) % num_days

        distribution = []
        start_idx = 0

        for day in range(num_days):
            # Первые дни получают на одного клиента больше
            clients_this_day = base_per_day + (1 if day < extra_clients else 0)

            day_clients = sorted_clients[start_idx:start_idx + clients_this_day]
            distribution.append(day_clients)

            print(f"  День {day + 1}: {len(day_clients)} клиентов")
            start_idx += clients_this_day

        return distribution

    def calculate_distance(self, client1: Client, client2: Client) -> float:
        """Вычисляет расстояние между клиентами (упрощенная версия)"""
        # Используем формулу гаверсинуса для приблизительного расстояния
        from math import radians, cos, sin, asin, sqrt

        lat1, lon1 = radians(client1.lat), radians(client1.lon)
        lat2, lon2 = radians(client2.lat), radians(client2.lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))

        # Радиус Земли в метрах
        r = 6371000
        return c * r

    def optimize_route_for_day(self, clients: List[Client], day: int) -> Route:
        """Оптимизирует маршрут для одного дня"""
        print(f"🗺️ Оптимизация маршрута для дня {day + 1} ({len(clients)} клиентов)...")

        if not clients:
            return Route(day=day, clients=[], total_distance=0, total_time=0, estimated_completion="00:00")

        # Простая оптимизация: начинаем с VIP клиентов, потом ближайшие
        optimized_clients = []
        remaining_clients = clients.copy()

        # Находим стартовую точку (центр всех клиентов)
        center_lat = sum(c.lat for c in clients) / len(clients)
        center_lon = sum(c.lon for c in clients) / len(clients)

        # Начинаем с ближайшего к центру клиента
        start_client = min(clients, key=lambda c:
            ((c.lat - center_lat)**2 + (c.lon - center_lon)**2)**0.5)

        optimized_clients.append(start_client)
        remaining_clients.remove(start_client)

        # Жадный алгоритм: каждый раз выбираем ближайшего
        current_client = start_client
        total_distance = 0
        total_time = 0

        while remaining_clients:
            # Находим ближайшего клиента
            next_client = min(remaining_clients, key=lambda c:
                self.calculate_distance(current_client, c))

            distance = self.calculate_distance(current_client, next_client)
            service_time = next_client.service_time_minutes

            total_distance += distance
            total_time += service_time + (distance / 1000) * 2  # 2 минуты на км

            optimized_clients.append(next_client)
            remaining_clients.remove(next_client)
            current_client = next_client

        # Оцениваем время завершения (начинаем в 9:00)
        start_hour = 9.0  # 9:00 утра
        estimated_hours = start_hour + (total_time / 60)
        estimated_completion = f"{int(estimated_hours):02d}:{int((estimated_hours % 1) * 60):02d}"

        return Route(
            day=day,
            clients=optimized_clients,
            total_distance=total_distance,
            total_time=total_time,
            estimated_completion=estimated_completion
        )

    def create_alternative_routes(self, clients: List[Client], day: int) -> Tuple[Route, Route]:
        """Создает 2 альтернативных маршрута для выбора"""
        print(f"🔄 Создание альтернативных маршрутов для дня {day + 1}...")

        # Маршрут 1: Приоритет VIP клиентам
        vip_clients = [c for c in clients if c.client_level == ClientLevel.VIP]
        regular_clients = [c for c in clients if c.client_level == ClientLevel.REGULAR]

        route1_clients = vip_clients + regular_clients
        route1 = self.optimize_route_for_day(route1_clients, day)

        # Маршрут 2: Географическая оптимизация (ближайшие соседи)
        route2_clients = clients.copy()
        # Сортируем по координатам для географической группировки
        route2_clients.sort(key=lambda c: (c.lat, c.lon))
        route2 = self.optimize_route_for_day(route2_clients, day)

        return route1, route2

    def plan_multi_day_routes(self, clients: List[Client], num_days: int) -> Dict:
        """Планирует маршруты на несколько дней с альтернативами"""
        print(f"📋 Планирование маршрутов на {num_days} дней...")

        # Распределяем клиентов по дням
        daily_clients = self.distribute_clients_by_days(clients, num_days)

        result = {
            'total_clients': len(clients),
            'num_days': num_days,
            'distribution': [],
            'routes': []
        }

        for day, day_clients in enumerate(daily_clients):
            print(f"\n📅 День {day + 1}: {len(day_clients)} клиентов")

            # Создаем альтернативные маршруты
            route1, route2 = self.create_alternative_routes(day_clients, day)

            day_info = {
                'day': day + 1,
                'num_clients': len(day_clients),
                'route_1': {
                    'clients': [c.id for c in route1.clients],
                    'total_distance': route1.total_distance,
                    'total_time': route1.total_time,
                    'estimated_completion': route1.estimated_completion,
                    'strategy': 'VIP Priority'
                },
                'route_2': {
                    'clients': [c.id for c in route2.clients],
                    'total_distance': route2.total_distance,
                    'total_time': route2.total_time,
                    'estimated_completion': route2.estimated_completion,
                    'strategy': 'Geographic Optimization'
                }
            }

            result['distribution'].append(day_info)
            result['routes'].extend([route1, route2])

        return result

    def mark_client_visited(self, client_id: int, actual_service_time: Optional[float] = None):
        """
        Отмечает клиента как посещенного и пересчитывает маршрут
        """
        print(f"✅ Клиент {client_id} отмечен как посещенный")

        # Добавляем в список посещенных
        self.visited_clients.add(client_id)

        # Обновляем текущее время
        if actual_service_time:
            self.current_time += actual_service_time / 60.0  # Переводим в часы
        else:
            # Используем стандартное время обслуживания
            client = next((c for c in self.clients if c.id == client_id), None)
            if client:
                self.current_time += client.service_time_minutes / 60.0

        print(f"🕐 Текущее время: {int(self.current_time):02d}:{int((self.current_time % 1) * 60):02d}")

        # Пересчитываем оставшиеся маршруты
        self.recalculate_remaining_routes()

    def recalculate_remaining_routes(self):
        """Пересчитывает оставшиеся маршруты с учетом посещенных клиентов"""
        print("🔄 Пересчет оставшихся маршрутов...")

        for day, route in self.current_routes.items():
            # Убираем посещенных клиентов
            remaining_clients = [c for c in route.clients if c.id not in self.visited_clients]

            if remaining_clients:
                # Пересчитываем маршрут для оставшихся клиентов
                new_route = self.optimize_route_for_day(remaining_clients, day)
                self.current_routes[day] = new_route

                print(f"  📅 День {day + 1}: {len(remaining_clients)} клиентов осталось")
                print(f"    ⏱️ Новое время завершения: {new_route.estimated_completion}")
            else:
                print(f"  📅 День {day + 1}: Все клиенты посещены ✅")

    def check_working_hours(self, client: Client, arrival_time: float) -> bool:
        """
        Проверяет, можно ли посетить клиента в указанное время
        """
        # Проверяем рабочие часы
        if arrival_time < client.work_start_hour or arrival_time > client.work_end_hour:
            return False

        # Проверяем обеденное время
        if client.lunch_start_hour <= arrival_time <= client.lunch_end_hour:
            return False

        return True

    def find_optimal_visit_time(self, client: Client, current_time: float) -> float:
        """
        Находит оптимальное время для посещения клиента
        """
        # Если сейчас можно посетить - посещаем
        if self.check_working_hours(client, current_time):
            return current_time

        # Иначе ищем ближайшее доступное время
        # Проверяем время после обеда
        if current_time < client.lunch_end_hour:
            return client.lunch_end_hour

        # Проверяем следующий рабочий день
        if current_time > client.work_end_hour:
            return client.work_start_hour + 24  # Следующий день

        return current_time

    def get_route_status(self) -> Dict:
        """Возвращает текущий статус маршрутов"""
        status = {
            'current_time': f"{int(self.current_time):02d}:{int((self.current_time % 1) * 60):02d}",
            'visited_clients': list(self.visited_clients),
            'remaining_routes': {}
        }

        for day, route in self.current_routes.items():
            remaining_clients = [c for c in route.clients if c.id not in self.visited_clients]
            status['remaining_routes'][f'day_{day + 1}'] = {
                'remaining_clients': len(remaining_clients),
                'estimated_completion': route.estimated_completion
            }

        return status

    def get_routes_json(self, clients: List[Client], num_days: int, include_waypoints: bool = True) -> Dict:
        """Возвращает маршруты в формате JSON для API"""
        result = self.plan_multi_day_routes(clients, num_days)

        # Конвертируем в JSON-совместимый формат
        json_result = {
            'success': True,
            'total_clients': result['total_clients'],
            'num_days': result['num_days'],
            'routes': []
        }

        for day_info in result['distribution']:
            # Получаем координаты для маршрута 1
            route1_waypoints = []
            if include_waypoints:
                for client_id in day_info['route_1']['clients']:
                    client = next((c for c in clients if c.id == client_id), None)
                    if client:
                        route1_waypoints.append({
                            'lat': client.lat,
                            'lon': client.lon,
                            'id': client.id,
                            'level': client.client_level.value
                        })

            # Получаем координаты для маршрута 2
            route2_waypoints = []
            if include_waypoints:
                for client_id in day_info['route_2']['clients']:
                    client = next((c for c in clients if c.id == client_id), None)
                    if client:
                        route2_waypoints.append({
                            'lat': client.lat,
                            'lon': client.lon,
                            'id': client.id,
                            'level': client.client_level.value
                        })

            day_routes = {
                'day': day_info['day'],
                'num_clients': day_info['num_clients'],
                'alternatives': [
                    {
                        'id': 1,
                        'strategy': day_info['route_1']['strategy'],
                        'clients': day_info['route_1']['clients'],
                        'waypoints': route1_waypoints,
                        'total_distance_meters': int(day_info['route_1']['total_distance']),
                        'total_time_minutes': int(day_info['route_1']['total_time']),
                        'estimated_completion': day_info['route_1']['estimated_completion']
                    },
                    {
                        'id': 2,
                        'strategy': day_info['route_2']['strategy'],
                        'clients': day_info['route_2']['clients'],
                        'waypoints': route2_waypoints,
                        'total_distance_meters': int(day_info['route_2']['total_distance']),
                        'total_time_minutes': int(day_info['route_2']['total_time']),
                        'estimated_completion': day_info['route_2']['estimated_completion']
                    }
                ]
            }
            json_result['routes'].append(day_routes)

        return json_result

    def get_tomtom_route_json(self, waypoints: List[Dict]) -> Dict:
        """Получает детальный маршрут от TomTom API"""
        if not self.tomtom_api_key:
            return {'error': 'TomTom API key not provided'}

        try:
            # Формируем URL для TomTom API
            coords_str = ':'.join([f"{wp['lat']},{wp['lon']}" for wp in waypoints])
            url = f"https://api.tomtom.com/routing/1/calculateRoute/{coords_str}/json"

            params = {
                'key': self.tomtom_api_key,
                'routeType': 'fastest',
                'traffic': 'true'
            }

            # Создаем сессию с SSL настройками
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

    def mark_visited_json(self, client_id: int, actual_service_time: Optional[float] = None) -> Dict:
        """Отмечает клиента как посещенного и возвращает обновленный статус в JSON"""
        self.mark_client_visited(client_id, actual_service_time)

        return {
            'success': True,
            'client_id': client_id,
            'status': self.get_route_status()
        }

def main():
    """Демонстрация системы оптимизации маршрутов"""
    print("🚀 Система оптимизации маршрутов")
    print("=" * 50)

    # Инициализируем оптимизатор с TomTom API
    optimizer = RouteOptimizer(tomtom_api_key="N0e11R91bFHexBDVlfIzDr7gjLygvdjv")

    # Загружаем клиентов
    clients = optimizer.load_clients_from_file("DATA (2).txt")
    if not clients:
        print("❌ Не удалось загрузить клиентов")
        return

    print(f"✅ Загружено {len(clients)} клиентов")
    print(f"👑 VIP клиентов: {sum(1 for c in clients if c.client_level == ClientLevel.VIP)}")
    print(f"👤 Обычных клиентов: {sum(1 for c in clients if c.client_level == ClientLevel.REGULAR)}")

    # Планируем маршруты на 3 дня
    num_days = 3
    result = optimizer.plan_multi_day_routes(clients, num_days)

    # Сохраняем маршруты для динамического перерасчета
    optimizer.current_routes = {i: result['routes'][i*2] for i in range(num_days)}
    optimizer.clients = clients

    # Выводим результаты
    print(f"\n📊 РЕЗУЛЬТАТЫ ПЛАНИРОВАНИЯ:")
    print("=" * 50)

    for day_info in result['distribution']:
        print(f"\n📅 День {day_info['day']}: {day_info['num_clients']} клиентов")
        print(f"  🛣️ Маршрут 1 ({day_info['route_1']['strategy']}):")
        print(f"    📏 Расстояние: {day_info['route_1']['total_distance']:.0f} м")
        print(f"    ⏱️ Время: {day_info['route_1']['total_time']:.0f} мин")
        print(f"    🏁 Завершение: {day_info['route_1']['estimated_completion']}")

        print(f"  🛣️ Маршрут 2 ({day_info['route_2']['strategy']}):")
        print(f"    📏 Расстояние: {day_info['route_2']['total_distance']:.0f} м")
        print(f"    ⏱️ Время: {day_info['route_2']['total_time']:.0f} мин")
        print(f"    🏁 Завершение: {day_info['route_2']['estimated_completion']}")

    # Демонстрация динамического перерасчета
    print(f"\n🔄 ДЕМОНСТРАЦИЯ ДИНАМИЧЕСКОГО ПЕРЕРАСЧЕТА:")
    print("=" * 50)

    # Симулируем посещение клиентов
    print("\n📋 Начальный статус:")
    status = optimizer.get_route_status()
    print(f"  🕐 Текущее время: {status['current_time']}")
    print(f"  👥 Посещенных клиентов: {len(status['visited_clients'])}")

    # Посещаем первого клиента
    first_client = result['routes'][0].clients[0]
    print(f"\n✅ Посещаем клиента {first_client.id} ({first_client.client_level.value})")
    optimizer.mark_client_visited(first_client.id, 25)  # 25 минут вместо стандартных 20/30

    # Посещаем второго клиента
    if len(result['routes'][0].clients) > 1:
        second_client = result['routes'][0].clients[1]
        print(f"\n✅ Посещаем клиента {second_client.id} ({second_client.client_level.value})")
        optimizer.mark_client_visited(second_client.id, 15)  # 15 минут вместо стандартных 20/30

    # Финальный статус
    print(f"\n📋 Финальный статус:")
    final_status = optimizer.get_route_status()
    print(f"  🕐 Текущее время: {final_status['current_time']}")
    print(f"  👥 Посещенных клиентов: {len(final_status['visited_clients'])}")

    for day_key, day_status in final_status['remaining_routes'].items():
        print(f"  📅 {day_key}: {day_status['remaining_clients']} клиентов осталось")

    # Демонстрация JSON API
    print(f"\n📡 ДЕМОНСТРАЦИЯ JSON API:")
    print("=" * 50)

    # Получаем маршруты в JSON формате с координатами
    json_routes = optimizer.get_routes_json(clients, num_days, include_waypoints=True)
    print("📋 JSON маршруты с координатами:")
    print(json.dumps(json_routes, indent=2, ensure_ascii=False)[:800] + "...")

    # Демонстрация TomTom API для детального маршрута
    print(f"\n🗺️ ДЕМОНСТРАЦИЯ TOMTOM API:")
    print("=" * 50)

    # Берем первые 3 точки для тестирования
    first_route = json_routes['routes'][0]['alternatives'][0]
    test_waypoints = first_route['waypoints'][:3]  # Первые 3 точки

    print(f"📍 Тестовые точки: {len(test_waypoints)}")
    for i, wp in enumerate(test_waypoints):
        print(f"  {i+1}. ID: {wp['id']}, Lat: {wp['lat']}, Lon: {wp['lon']}, Level: {wp['level']}")

    # Получаем детальный маршрут от TomTom
    tomtom_result = optimizer.get_tomtom_route_json(test_waypoints)

    if 'error' in tomtom_result:
        print(f"❌ TomTom API ошибка: {tomtom_result['error']}")
    else:
        print(f"✅ TomTom API успешно:")
        print(f"  📏 Расстояние: {tomtom_result['routes'][0]['summary']['lengthInMeters']} м")
        print(f"  ⏱️ Время: {tomtom_result['routes'][0]['summary']['travelTimeInSeconds']} сек")
        print(f"  🚦 Пробки: {tomtom_result['routes'][0]['summary']['trafficDelayInSeconds']} сек")

    # Демонстрация отметки клиента в JSON
    print(f"\n✅ Отметка клиента в JSON:")
    mark_result = optimizer.mark_visited_json(35, 25)
    print(json.dumps(mark_result, indent=2, ensure_ascii=False))

    print(f"\n✅ Планирование и демонстрация завершены!")
    return result

if __name__ == "__main__":
    main()
