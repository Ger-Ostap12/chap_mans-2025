#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📍 Модуль определения местоположения пользователя
"""

import requests
import json
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class Location:
    """Местоположение пользователя"""
    latitude: float
    longitude: float
    address: str
    city: str
    country: str
    accuracy: float  # Точность в метрах
    source: str  # GPS, IP, Manual

class LocationDetector:
    """Детектор местоположения пользователя"""

    def __init__(self):
        self.default_location = Location(
            latitude=47.2225,  # Ростов-на-Дону
            longitude=39.7203,
            address="Ростов-на-Дону, Россия",
            city="Ростов-на-Дону",
            country="Россия",
            accuracy=1000.0,
            source="default"
        )

    def get_location_from_gps(self, lat: float, lon: float) -> Location:
        """Получает местоположение по GPS координатам"""
        print(f"📍 Получение местоположения по GPS: {lat}, {lon}")

        try:
            # Обратное геокодирование через OpenStreetMap Nominatim
            url = f"https://nominatim.openstreetmap.org/reverse"
            params = {
                'lat': lat,
                'lon': lon,
                'format': 'json',
                'addressdetails': 1
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                address_parts = data.get('address', {})
                city = address_parts.get('city', address_parts.get('town', 'Неизвестно'))
                country = address_parts.get('country', 'Неизвестно')

                location = Location(
                    latitude=lat,
                    longitude=lon,
                    address=data.get('display_name', 'Неизвестный адрес'),
                    city=city,
                    country=country,
                    accuracy=10.0,  # GPS точность
                    source="GPS"
                )

                print(f"✅ GPS местоположение: {location.address}")
                return location
            else:
                print(f"❌ Ошибка GPS геокодирования: {response.status_code}")
                return self._create_location_from_coords(lat, lon, "GPS")

        except Exception as e:
            print(f"❌ Ошибка GPS определения: {e}")
            return self._create_location_from_coords(lat, lon, "GPS")

    def get_location_from_ip(self, ip_address: str = None) -> Location:
        """Получает местоположение по IP адресу"""
        print(f"🌐 Получение местоположения по IP: {ip_address or 'автоматически'}")

        try:
            # Используем ipapi.co для определения местоположения по IP
            if ip_address:
                url = f"https://ipapi.co/{ip_address}/json/"
            else:
                url = "https://ipapi.co/json/"

            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                location = Location(
                    latitude=float(data.get('latitude', 47.2225)),
                    longitude=float(data.get('longitude', 39.7203)),
                    address=f"{data.get('city', 'Неизвестно')}, {data.get('country_name', 'Неизвестно')}",
                    city=data.get('city', 'Неизвестно'),
                    country=data.get('country_name', 'Неизвестно'),
                    accuracy=5000.0,  # IP точность
                    source="IP"
                )

                print(f"✅ IP местоположение: {location.address}")
                return location
            else:
                print(f"❌ Ошибка IP геолокации: {response.status_code}")
                return self.default_location

        except Exception as e:
            print(f"❌ Ошибка IP определения: {e}")
            return self.default_location

    def get_location_from_address(self, address: str) -> Location:
        """Получает местоположение по адресу (геокодирование)"""
        print(f"🏠 Получение местоположения по адресу: {address}")

        try:
            # Прямое геокодирование через OpenStreetMap Nominatim
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': address,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if data:
                    result = data[0]

                    location = Location(
                        latitude=float(result.get('lat', 47.2225)),
                        longitude=float(result.get('lon', 39.7203)),
                        address=result.get('display_name', address),
                        city=result.get('address', {}).get('city', 'Неизвестно'),
                        country=result.get('address', {}).get('country', 'Неизвестно'),
                        accuracy=100.0,  # Адресная точность
                        source="Manual"
                    )

                    print(f"✅ Адресное местоположение: {location.address}")
                    return location
                else:
                    print(f"❌ Адрес не найден: {address}")
                    return self.default_location
            else:
                print(f"❌ Ошибка адресного геокодирования: {response.status_code}")
                return self.default_location

        except Exception as e:
            print(f"❌ Ошибка адресного определения: {e}")
            return self.default_location

    def _create_location_from_coords(self, lat: float, lon: float, source: str) -> Location:
        """Создает объект местоположения из координат"""
        return Location(
            latitude=lat,
            longitude=lon,
            address=f"Координаты: {lat:.6f}, {lon:.6f}",
            city="Неизвестно",
            country="Неизвестно",
            accuracy=100.0,
            source=source
        )

    def get_best_location(self, gps_coords: Optional[Tuple[float, float]] = None,
                         ip_address: Optional[str] = None,
                         manual_address: Optional[str] = None) -> Location:
        """Получает лучшее доступное местоположение"""
        print("📍 Определение лучшего местоположения...")

        # Приоритет: GPS > Manual > IP > Default
        if gps_coords:
            lat, lon = gps_coords
            return self.get_location_from_gps(lat, lon)

        if manual_address:
            return self.get_location_from_address(manual_address)

        if ip_address:
            return self.get_location_from_ip(ip_address)

        # Пробуем автоматическое IP определение
        try:
            return self.get_location_from_ip()
        except:
            print("⚠️ Не удалось определить местоположение, используем по умолчанию")
            return self.default_location

    def validate_location(self, location: Location) -> bool:
        """Проверяет валидность местоположения"""
        if not location:
            return False

        # Проверяем координаты (примерно для России)
        if not (40.0 <= location.latitude <= 80.0 and 20.0 <= location.longitude <= 180.0):
            print(f"⚠️ Координаты вне разумных пределов: {location.latitude}, {location.longitude}")
            return False

        return True

def main():
    """Тестирование модуля определения местоположения"""
    print("📍 Тестирование модуля определения местоположения")
    print("=" * 60)

    detector = LocationDetector()

    # Тест 1: GPS координаты
    print("\n1️⃣ Тест GPS координат:")
    gps_location = detector.get_location_from_gps(47.2225, 39.7203)
    print(f"   📍 {gps_location.address}")
    print(f"   🎯 Точность: {gps_location.accuracy}м")
    print(f"   📡 Источник: {gps_location.source}")

    # Тест 2: IP геолокация
    print("\n2️⃣ Тест IP геолокации:")
    ip_location = detector.get_location_from_ip()
    print(f"   📍 {ip_location.address}")
    print(f"   🎯 Точность: {ip_location.accuracy}м")
    print(f"   📡 Источник: {ip_location.source}")

    # Тест 3: Адресное геокодирование
    print("\n3️⃣ Тест адресного геокодирования:")
    address_location = detector.get_location_from_address("Ростов-на-Дону, Театральная площадь")
    print(f"   📍 {address_location.address}")
    print(f"   🎯 Точность: {address_location.accuracy}м")
    print(f"   📡 Источник: {address_location.source}")

    # Тест 4: Лучшее местоположение
    print("\n4️⃣ Тест лучшего местоположения:")
    best_location = detector.get_best_location(
        gps_coords=(47.2225, 39.7203),
        manual_address="Ростов-на-Дону"
    )
    print(f"   📍 {best_location.address}")
    print(f"   🎯 Точность: {best_location.accuracy}м")
    print(f"   📡 Источник: {best_location.source}")

    print("\n✅ Тестирование завершено!")

if __name__ == "__main__":
    main()
