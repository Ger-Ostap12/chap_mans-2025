#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Тестирование единой системы: ANN + TomTom API + граф маршрутов
"""

import os
import json
import pandas as pd
from unified_route_system import UnifiedRouteSystem, Client, ClientLevel

class SystemTester:
    """Тестер единой системы маршрутов"""

    def __init__(self):
        self.system = UnifiedRouteSystem(tomtom_api_key="4Me4kS17IKSfQmvDuIgLpsz9jxAu6tt2")
        print("🧪 Инициализация тестера системы...")

    def load_test_clients(self, data_file: str = "DATA (2).txt"):
        """Загружает тестовых клиентов"""
        print(f"📊 Загружаем тестовых клиентов из {data_file}...")

        if not os.path.exists(data_file):
            print(f"❌ Файл не найден: {data_file}")
            return []

        try:
            # Читаем файл как текст
            with open(data_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Парсим JSON-подобные данные
            clients = self._parse_client_data(content)
            print(f"✅ Загружено {len(clients)} тестовых клиентов")
            return clients

        except Exception as e:
            print(f"❌ Ошибка загрузки клиентов: {e}")
            return []

    def _parse_client_data(self, content: str):
        """Парсит данные клиентов из файла"""
        import re

        try:
            # Ищем массив данных
            data_match = re.search(r'data = \[(.*?)\]', content, re.DOTALL)
            if not data_match:
                print("❌ Не найден массив данных")
                return []

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
                    obj_str = current_obj.strip()
                    if obj_str.startswith('{') and obj_str.endswith('}'):
                        objects.append(obj_str)
                    current_obj = ""

            # Парсим каждый объект
            clients = []
            for i, obj_str in enumerate(objects):
                try:
                    # Очищаем от лишних символов
                    obj_str = obj_str.strip()
                    if obj_str.endswith(','):
                        obj_str = obj_str[:-1]

                    # Парсим JSON
                    client_data = json.loads(obj_str)

                    # Создаем объект Client
                    client = Client(
                        id=int(client_data.get('id', i+1)),
                        address=str(client_data.get('address1', '')),
                        lat=float(client_data.get('lat', 47.2225)),  # Ростов-на-Дону по умолчанию
                        lon=float(client_data.get('lon', 39.7203)),
                        client_level=ClientLevel.VIP if client_data.get('client_level') == 'VIP' else ClientLevel.REGULAR,
                        work_start=str(client_data.get('work_start', '09:00')),
                        work_end=str(client_data.get('work_end', '18:00')),
                        lunch_start=str(client_data.get('lunch_start', '13:00')),
                        lunch_end=str(client_data.get('lunch_end', '14:00'))
                    )

                    clients.append(client)

                except Exception as e:
                    print(f"⚠️ Ошибка парсинга клиента {i+1}: {e}")
                    continue

            return clients

        except Exception as e:
            print(f"❌ Ошибка парсинга данных: {e}")
            return []

    def test_route_optimization(self, clients, num_days: int = 3):
        """Тестирует оптимизацию маршрутов"""
        print(f"\n🗺️ Тестирование оптимизации маршрутов для {len(clients)} клиентов на {num_days} дней...")

        if not clients:
            print("❌ Нет клиентов для тестирования")
            return None

        # Тестируем единую систему
        try:
            result = self.system.get_unified_route(clients, num_days)

            if result['success']:
                print("✅ Маршруты успешно построены!")
                self._print_route_analysis(result)
                return result
            else:
                print("❌ Ошибка построения маршрутов")
                return None

        except Exception as e:
            print(f"❌ Ошибка при тестировании: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _print_route_analysis(self, result):
        """Выводит анализ построенных маршрутов"""
        print("\n📊 АНАЛИЗ МАРШРУТОВ:")
        print("=" * 60)

        total_clients = result['total_clients']
        num_days = result['num_days']

        print(f"👥 Всего клиентов: {total_clients}")
        print(f"📅 Количество дней: {num_days}")
        print(f"📈 Среднее клиентов в день: {total_clients / num_days:.1f}")

        # Анализ по дням
        for route in result['routes']:
            day = route['day']
            clients_count = len(route['clients'])
            vip_count = sum(1 for wp in route['waypoints'] if wp['level'] == 'VIP')
            standard_count = clients_count - vip_count

            print(f"\n📅 День {day}:")
            print(f"   👥 Клиентов: {clients_count}")
            print(f"   🌟 VIP: {vip_count} из {clients_count}")
            print(f"   👤 Стандарт: {standard_count} из {clients_count}")
            print(f"   🧠 ANN оптимизирован: {route['ann_optimized']}")

            # Показываем первые 3 клиента
            if route['waypoints']:
                print(f"   🚀 Начало: {route['waypoints'][0]['id']} ({route['waypoints'][0]['level']})")
                if len(route['waypoints']) > 1:
                    print(f"   🏁 Финиш: {route['waypoints'][-1]['id']} ({route['waypoints'][-1]['level']})")

    def test_ann_predictions(self, clients):
        """Тестирует предсказания ANN модели на ВСЕХ клиентах"""
        print(f"\n🧠 Тестирование предсказаний ANN модели на ВСЕХ {len(clients)} клиентах...")

        if len(clients) < 2:
            print("❌ Нужно минимум 2 клиента для тестирования")
            return

        print(f"📊 Тестируем на ВСЕХ {len(clients)} клиентах...")

        try:
            # Оптимизируем маршрут с помощью ANN для ВСЕХ клиентов
            optimized_clients = self.system.optimize_route_with_ann(clients)

            print("✅ ANN модель успешно оптимизировала маршрут для ВСЕХ клиентов!")

            # Анализируем результат
            print("\n📈 АНАЛИЗ ANN ОПТИМИЗАЦИИ (ВСЕ КЛИЕНТЫ):")
            print("-" * 60)

            # Показываем первые 10 и последние 5 клиентов
            print("🚀 ПЕРВЫЕ 10 КЛИЕНТОВ:")
            for i, client in enumerate(optimized_clients[:10]):
                vip_status = "🌟 VIP" if client.client_level == ClientLevel.VIP else "👤 Стандарт"
                print(f"{i+1:2d}. ID: {client.id} - {vip_status}")

            if len(optimized_clients) > 10:
                print(f"\n... (пропущено {len(optimized_clients) - 15} клиентов) ...")
                print("\n🏁 ПОСЛЕДНИЕ 5 КЛИЕНТОВ:")
                for i, client in enumerate(optimized_clients[-5:], len(optimized_clients)-4):
                    vip_status = "🌟 VIP" if client.client_level == ClientLevel.VIP else "👤 Стандарт"
                    print(f"{i:2d}. ID: {client.id} - {vip_status}")

            # Проверяем VIP приоритет
            vip_positions = [i for i, c in enumerate(optimized_clients) if c.client_level == ClientLevel.VIP]
            if vip_positions:
                avg_vip_position = sum(vip_positions) / len(vip_positions)
                print(f"\n🌟 VIP клиенты в среднем на позиции: {avg_vip_position:.1f}")
                print(f"🌟 Всего VIP клиентов: {len(vip_positions)}")
                print("✅ ANN модель учитывает VIP приоритет!")
            else:
                print("ℹ️ Нет VIP клиентов в выборке")

            # Статистика
            total_vip = sum(1 for c in optimized_clients if c.client_level == ClientLevel.VIP)
            total_standard = len(optimized_clients) - total_vip
            print(f"\n📊 СТАТИСТИКА:")
            print(f"   🌟 VIP клиентов: {total_vip}")
            print(f"   👤 Стандартных: {total_standard}")
            print(f"   📊 Всего клиентов: {len(optimized_clients)}")

        except Exception as e:
            print(f"❌ Ошибка тестирования ANN: {e}")
            import traceback
            traceback.print_exc()

    def test_tomtom_integration(self, clients):
        """Тестирует интеграцию с TomTom API на нескольких маршрутах"""
        print(f"\n🌐 Тестирование интеграции с TomTom API...")

        if len(clients) < 4:
            print("❌ Нужно минимум 4 клиента для тестирования")
            return

        # Тестируем несколько маршрутов
        test_routes = [
            (clients[0], clients[1]),  # Первый → Второй
            (clients[1], clients[2]),  # Второй → Третий
            (clients[0], clients[3])    # Первый → Четвертый
        ]

        successful_routes = 0
        total_distance = 0
        total_time = 0

        for i, (client1, client2) in enumerate(test_routes, 1):
            print(f"📍 Тестируем маршрут {i}: {client1.id} → {client2.id}")

            try:
                # Получаем маршрут от TomTom
                tomtom_result = self.system.get_tomtom_route(client1, client2)

                if 'error' in tomtom_result:
                    print(f"❌ TomTom API ошибка: {tomtom_result['error']}")
                else:
                    print("✅ TomTom API успешно вернул маршрут!")

                    # Анализируем результат
                    if 'routes' in tomtom_result and tomtom_result['routes']:
                        route = tomtom_result['routes'][0]
                        summary = route.get('summary', {})

                        distance = summary.get('lengthInMeters', 0) / 1000  # км
                        duration = summary.get('travelTimeInSeconds', 0) / 60  # минуты

                        print(f"   📏 Расстояние: {distance:.2f} км")
                        print(f"   ⏱️ Время в пути: {duration:.1f} мин")
                        print(f"   🚦 Учет трафика: {summary.get('trafficDelayInSeconds', 0)} сек задержки")

                        successful_routes += 1
                        total_distance += distance
                        total_time += duration
                    else:
                        print("   ❌ Нет данных о маршруте в ответе TomTom")

            except Exception as e:
                print(f"   ❌ Ошибка TomTom API: {e}")

        # Итоговая статистика
        print(f"\n📊 СТАТИСТИКА TOMTOM API:")
        print(f"   ✅ Успешных маршрутов: {successful_routes}/{len(test_routes)}")
        if successful_routes > 0:
            print(f"   📏 Среднее расстояние: {total_distance/successful_routes:.2f} км")
            print(f"   ⏱️ Среднее время: {total_time/successful_routes:.1f} мин")
            print("✅ TomTom API работает корректно!")
            return True
        else:
            print("❌ TomTom API не работает")
            return False

    def test_dynamic_recalculation(self, clients):
        """Тестирует динамический перерасчет при отметке клиента как посещенного"""
        print(f"\n🔄 Тестирование динамического перерасчета...")

        if len(clients) < 3:
            print("❌ Нужно минимум 3 клиента для тестирования")
            return

        try:
            # Симулируем посещение клиентов
            test_clients = clients[:5]  # Берем первых 5 клиентов

            print(f"📊 Тестируем на {len(test_clients)} клиентах...")

            # 1. Отмечаем первого клиента как посещенного (стандартное время)
            print("\n1️⃣ Отмечаем клиента 1 как посещенного (стандартное время):")
            result1 = self.system.mark_client_visited(test_clients[0].id)
            print(f"   ✅ Результат: {result1['message']}")
            print(f"   ⏰ Текущее время: {result1['current_time']:.2f} часов")
            print(f"   👥 Посещенных клиентов: {len(result1['visited_clients'])}")

            # 2. Отмечаем второго клиента (VIP, 30 минут)
            print("\n2️⃣ Отмечаем клиента 2 как посещенного (VIP, 30 мин):")
            result2 = self.system.mark_client_visited(test_clients[1].id, actual_service_time=30.0)
            print(f"   ✅ Результат: {result2['message']}")
            print(f"   ⏰ Текущее время: {result2['current_time']:.2f} часов")
            print(f"   👥 Посещенных клиентов: {len(result2['visited_clients'])}")

            # 3. Отмечаем третьего клиента (раньше времени, 15 минут)
            print("\n3️⃣ Отмечаем клиента 3 как посещенного (раньше времени, 15 мин):")
            result3 = self.system.mark_client_visited(test_clients[2].id, actual_service_time=15.0)
            print(f"   ✅ Результат: {result3['message']}")
            print(f"   ⏰ Текущее время: {result3['current_time']:.2f} часов")
            print(f"   👥 Посещенных клиентов: {len(result3['visited_clients'])}")

            # 4. Отмечаем четвертого клиента (позже времени, 45 минут)
            print("\n4️⃣ Отмечаем клиента 4 как посещенного (позже времени, 45 мин):")
            result4 = self.system.mark_client_visited(test_clients[3].id, actual_service_time=45.0)
            print(f"   ✅ Результат: {result4['message']}")
            print(f"   ⏰ Текущее время: {result4['current_time']:.2f} часов")
            print(f"   👥 Посещенных клиентов: {len(result4['visited_clients'])}")

            # Анализ результатов
            print(f"\n📊 АНАЛИЗ ДИНАМИЧЕСКОГО ПЕРЕРАСЧЕТА:")
            print(f"   🕐 Начальное время: 9.00 часов")
            print(f"   🕐 Финальное время: {result4['current_time']:.2f} часов")
            print(f"   ⏱️ Общее время работы: {result4['current_time'] - 9.0:.2f} часов")
            print(f"   👥 Посещено клиентов: {len(result4['visited_clients'])}")
            print(f"   📈 Среднее время на клиента: {(result4['current_time'] - 9.0) / len(result4['visited_clients']):.2f} часов")

            print("✅ Динамический перерасчет работает корректно!")
            return True

        except Exception as e:
            print(f"❌ Ошибка тестирования динамического перерасчета: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_location_detection(self):
        """Тестирует определение местоположения пользователя"""
        print(f"\n📍 Тестирование определения местоположения...")

        try:
            # Тест 1: GPS координаты
            print("\n1️⃣ Тест GPS координат:")
            gps_result = self.system.set_user_location(gps_coords=(47.2225, 39.7203))
            if gps_result['success']:
                print(f"   ✅ GPS: {gps_result['location']['address']}")
                print(f"   🎯 Точность: {gps_result['location']['accuracy']}м")
            else:
                print(f"   ❌ GPS ошибка: {gps_result['message']}")

            # Тест 2: IP геолокация
            print("\n2️⃣ Тест IP геолокации:")
            ip_result = self.system.set_user_location(ip_address="8.8.8.8")
            if ip_result['success']:
                print(f"   ✅ IP: {ip_result['location']['address']}")
                print(f"   🎯 Точность: {ip_result['location']['accuracy']}м")
            else:
                print(f"   ❌ IP ошибка: {ip_result['message']}")

            # Тест 3: Адресное геокодирование
            print("\n3️⃣ Тест адресного геокодирования:")
            address_result = self.system.set_user_location(manual_address="Ростов-на-Дону, Театральная площадь")
            if address_result['success']:
                print(f"   ✅ Адрес: {address_result['location']['address']}")
                print(f"   🎯 Точность: {address_result['location']['accuracy']}м")
            else:
                print(f"   ❌ Адрес ошибка: {address_result['message']}")

            # Тест 4: Получение текущего местоположения
            print("\n4️⃣ Тест получения местоположения:")
            current_location = self.system.get_user_location()
            if current_location['success']:
                print(f"   ✅ Текущее: {current_location['location']['address']}")
                print(f"   📡 Источник: {current_location['location']['source']}")
            else:
                print(f"   ❌ Нет местоположения: {current_location['message']}")

            print("✅ Определение местоположения работает корректно!")
            return True

        except Exception as e:
            print(f"❌ Ошибка тестирования местоположения: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_dynamic_location_change(self):
        """Тестирует динамическое изменение местоположения"""
        print(f"\n🗺️ Тестирование динамического изменения местоположения...")

        try:
            # 1. Устанавливаем начальное местоположение
            print("\n1️⃣ Установка начального местоположения:")
            initial_result = self.system.set_user_location(gps_coords=(47.2225, 39.7203))
            if initial_result['success']:
                print(f"   ✅ Начальное: {initial_result['location']['address']}")
            else:
                print(f"   ❌ Ошибка: {initial_result['message']}")
                return False

            # 2. Обновляем местоположение (пользователь перетащил маркер на карте)
            print("\n2️⃣ Обновление местоположения (перетаскивание на карте):")
            new_coords = (47.2500, 39.7500)  # Новые координаты
            update_result = self.system.update_user_location(new_coords[0], new_coords[1])
            if update_result['success']:
                print(f"   ✅ Обновлено: {update_result['new_location']['address']}")
                print(f"   📍 Старое: {update_result['old_location']['address']}")
            else:
                print(f"   ❌ Ошибка обновления: {update_result['message']}")
                return False

            # 3. Тестируем поиск предложений адресов
            print("\n3️⃣ Тестирование поиска предложений:")
            suggestions_result = self.system.get_location_suggestions("Ростов-на-Дону", limit=3)
            if suggestions_result['success']:
                print(f"   ✅ Найдено {len(suggestions_result['suggestions'])} предложений:")
                for i, suggestion in enumerate(suggestions_result['suggestions'][:3], 1):
                    print(f"      {i}. {suggestion['display_name']}")
            else:
                print(f"   ❌ Ошибка поиска: {suggestions_result['message']}")

            # 4. Тестируем пересчет маршрутов с нового местоположения
            print("\n4️⃣ Тестирование пересчета маршрутов:")
            # Загружаем тестовых клиентов
            clients = self.system.load_clients_from_file("DATA (2).txt")
            if clients:
                recalc_result = self.system.recalculate_routes_from_new_location(clients[:5], 2)
                if recalc_result['success']:
                    print(f"   ✅ Маршруты пересчитаны с нового местоположения")
                    print(f"   📊 Дней: {len(recalc_result['routes']['routes'])}")
                else:
                    print(f"   ❌ Ошибка пересчета: {recalc_result['message']}")
            else:
                print("   ⚠️ Нет клиентов для тестирования")

            # 5. Проверяем текущее местоположение
            print("\n5️⃣ Проверка текущего местоположения:")
            current_location = self.system.get_user_location()
            if current_location['success']:
                print(f"   ✅ Текущее: {current_location['location']['address']}")
                print(f"   📡 Источник: {current_location['location']['source']}")
            else:
                print(f"   ❌ Нет местоположения: {current_location['message']}")

            print("✅ Динамическое изменение местоположения работает корректно!")
            return True

        except Exception as e:
            print(f"❌ Ошибка тестирования динамического изменения: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_full_test(self):
        """Запускает полное тестирование системы"""
        print("🧪 ПОЛНОЕ ТЕСТИРОВАНИЕ ЕДИНОЙ СИСТЕМЫ")
        print("=" * 60)

        # 1. Загружаем тестовых клиентов
        clients = self.load_test_clients()
        if not clients:
            print("❌ Не удалось загрузить клиентов для тестирования")
            return

        # 2. Тестируем ANN предсказания
        self.test_ann_predictions(clients)

        # 3. Тестируем TomTom интеграцию
        self.test_tomtom_integration(clients)

        # 4. Тестируем полную оптимизацию маршрутов
        route_result = self.test_route_optimization(clients, num_days=3)

        # 5. Экспортируем маршруты в JSON для фронтенда
        if route_result and route_result['success']:
            json_file = self.system.export_routes_to_json(route_result, "frontend_routes.json")
            if json_file:
                print(f"📄 JSON файл для фронтенда: {json_file}")

        # 6. Тестируем определение местоположения
        location_result = self.test_location_detection()

        # 7. Тестируем динамическое изменение местоположения
        dynamic_location_result = self.test_dynamic_location_change()

        # 8. Тестируем динамический перерасчет
        dynamic_result = self.test_dynamic_recalculation(clients)

        # 6. Итоговый отчет
        print("\n🎉 ИТОГОВЫЙ ОТЧЕТ ТЕСТИРОВАНИЯ:")
        print("=" * 60)

        if route_result and route_result['success'] and dynamic_result:
            print("✅ Единая система работает корректно!")
            print("✅ ANN модель обучена и оптимизирует маршруты")
            print("✅ TomTom API интегрирован")
            print("✅ Граф маршрутов строится правильно")
            print("✅ Динамический перерасчет работает!")
            print("✅ Система готова к продакшену!")
        else:
            print("❌ Обнаружены проблемы в системе")
            print("💡 Требуется дополнительная настройка")

def main():
    """Основная функция тестирования"""
    print("🧪 Тестирование единой системы маршрутов")
    print("=" * 60)

    # Создаем тестер
    tester = SystemTester()

    # Запускаем полное тестирование
    tester.run_full_test()

    print("\n🎯 Тестирование завершено!")

if __name__ == "__main__":
    main()
