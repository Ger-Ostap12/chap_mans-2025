import math
import json
import requests
import logging
import time
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YandexTrafficParser:
    def __init__(self):
        self.session = requests.Session()
        self.setup_headers()
        self.base_url = "https://yandex.ru/maps/"
        
    def setup_headers(self):
        """Настройка заголовков для имитации браузера"""
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        })

    def haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Расчет расстояния между точками в метрах"""
        R = 6371000  # радиус Земли в метрах
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def get_traffic_data_from_html(self, lat: float, lon: float, zoom: int = 12) -> Optional[Dict]:
        """
        Получение данных о пробках через парсинг HTML страницы Яндекс Карт
        """
        try:
            # Параметры для запроса
            params = {
                'll': f'{lon},{lat}',
                'z': zoom,
                'l': 'map,trf',  # trf - трафик
                'mode': 'search',
                'text': f'{lat},{lon}'
            }
            
            url = f"{self.base_url}?{urlencode(params)}"
            logger.info(f"Запрос к Яндекс Картам: {url}")
            
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                logger.error(f"Ошибка HTTP: {response.status_code}")
                return None
                
            return self.parse_traffic_from_html(response.text, lat, lon)
            
        except Exception as e:
            logger.error(f"Ошибка при получении данных: {e}")
            return None

    def parse_traffic_from_html(self, html: str, lat: float, lon: float) -> Optional[Dict]:
        """
        Парсинг HTML для извлечения данных о пробках
        """
        try:
            traffic_data = {
                'jams': [],
                'events': [],
                'overall_level': 0
            }
            
            # Поиск JSON данных в скриптах
            json_patterns = [
                r'window\.YMaps\.jams\s*=\s*({[^;]+});',
                r'"traffic":\s*({[^}]+})',
                r'jamsInfo\s*:\s*({[^}]+})',
                r'trafficInfo\s*:\s*({[^}]+})'
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, html)
                for match in matches:
                    try:
                        data = json.loads(match)
                        if self.validate_traffic_data(data):
                            return self.enrich_traffic_data(data, lat, lon)
                    except json.JSONDecodeError:
                        continue
            
            # Альтернативный метод: поиск по классам и атрибутам
            traffic_elements = re.findall(r'class="[^"]*traffic[^"]*"[^>]*data-json\s*=\s*"([^"]+)"', html)
            for element in traffic_elements:
                try:
                    # Декодирование HTML entities
                    import html # type: ignore
                    decoded_json = html.unescape(element) # type: ignore
                    data = json.loads(decoded_json)
                    if self.validate_traffic_data(data):
                        return self.enrich_traffic_data(data, lat, lon)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Ошибка парсинга элемента: {e}")
                    continue
            
            # Если не нашли структурированных данных, пытаемся извлечь из текста
            return self.extract_traffic_from_text(html, lat, lon)
            
        except Exception as e:
            logger.error(f"Ошибка парсинга HTML: {e}")
            return None

    def validate_traffic_data(self, data: Dict) -> bool:
        """Проверка валидности данных о трафике"""
        required_keys = ['jams', 'events']
        return any(key in data for key in required_keys)

    def enrich_traffic_data(self, data: Dict, lat: float, lon: float) -> Dict:
        """Обогащение данных о трафике координатами и расстояниями"""
        enriched_data = {
            'jams': [],
            'events': [],
            'overall_level': data.get('level', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        # Обработка пробок
        if 'jams' in data and isinstance(data['jams'], list):
            for jam in data['jams']:
                enriched_jam = self.process_jam_data(jam, lat, lon)
                if enriched_jam:
                    enriched_data['jams'].append(enriched_jam)
        
        # Обработка событий
        if 'events' in data and isinstance(data['events'], list):
            for event in data['events']:
                enriched_event = self.process_event_data(event, lat, lon)
                if enriched_event:
                    enriched_data['events'].append(enriched_event)
                    
        return enriched_data

    def extract_traffic_from_text(self, html: str, lat: float, lon: float) -> Dict:
        """
        Извлечение информации о пробках из текста страницы
        """
        traffic_data = {
            'jams': [],
            'events': [],
            'overall_level': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Поиск упоминаний о пробках в тексте
            traffic_patterns = [
                r'пробк[а-яё]*', r'затор[а-яё]*', r'traffic', r'jam',
                r'ДТП', r'авария', r'accident', r'ремонт', r'repair'
            ]
            
            for pattern in traffic_patterns:
                matches = re.findall(pattern, html, re.IGNORECASE)
                if matches:
                    # Создаем фиктивное событие на основе найденных ключевых слов
                    event = {
                        'id': f"text_{int(time.time())}",
                        'type': 'traffic_mention',
                        'title': 'Упоминание о пробках',
                        'description': f"Найдены упоминания: {', '.join(set(matches[:3]))}",
                        'position': [lon, lat],
                        'distance_m': 0,
                        'source': 'YandexMaps_HTML'
                    }
                    traffic_data['events'].append(event)
                    break
                    
        except Exception as e:
            logger.debug(f"Ошибка извлечения из текста: {e}")
            
        return traffic_data

    def process_jam_data(self, jam: Dict, center_lat: float, center_lon: float) -> Optional[Dict]:
        """Обработка данных о пробке"""
        try:
            # Получаем координаты пробки
            if 'geometry' in jam and 'coordinates' in jam['geometry']:
                coords = jam['geometry']['coordinates']
                if isinstance(coords, list) and len(coords) >= 2:
                    jam_lon, jam_lat = coords[0], coords[1]
                else:
                    return None
            else:
                return None
            
            distance = self.haversine(center_lat, center_lon, jam_lat, jam_lon)
            
            return {
                'id': f"jam_{jam.get('id', int(time.time()))}",
                'type': 'traffic_jam',
                'level': jam.get('level', 0),
                'speed': jam.get('speed', 0),
                'length': jam.get('length', 0),
                'position': [jam_lon, jam_lat],
                'distance_m': round(distance, 1),
                'description': f"Уровень: {jam.get('level', 0)}, Скорость: {jam.get('speed', 0)} км/ч"
            }
            
        except Exception as e:
            logger.debug(f"Ошибка обработки пробки: {e}")
            return None

    def process_event_data(self, event: Dict, center_lat: float, center_lon: float) -> Optional[Dict]:
        """Обработка данных о дорожном событии"""
        try:
            if 'point' in event and isinstance(event['point'], list) and len(event['point']) >= 2:
                event_lon, event_lat = event['point'][0], event['point'][1]
            else:
                return None
            
            distance = self.haversine(center_lat, center_lon, event_lat, event_lon)
            
            event_type = event.get('type', 'unknown')
            description = event.get('description', '')
            
            return {
                'id': f"event_{event.get('id', int(time.time()))}",
                'type': event_type,
                'title': self.get_event_title(event_type),
                'description': description,
                'position': [event_lon, event_lat],
                'distance_m': round(distance, 1),
                'source': 'YandexMaps'
            }
            
        except Exception as e:
            logger.debug(f"Ошибка обработки события: {e}")
            return None

    def get_event_title(self, event_type: str) -> str:
        """Получение заголовка для типа события"""
        titles = {
            'accident': 'ДТП',
            'road_works': 'Дорожные работы',
            'road_closure': 'Перекрытие дороги',
            'danger': 'Опасность',
            'police': 'Патруль ДПС',
            'traffic_jam': 'Пробка',
            'unknown': 'Дорожное событие'
        }
        return titles.get(event_type, 'Дорожное событие')

    def get_traffic_level_description(self, level: int) -> str:
        """Описание уровня пробок"""
        levels = {
            0: 'Свободно',
            1: 'Мало машин',
            2: 'Умеренное движение',
            3: 'Насыщенное движение',
            4: 'Пробка',
            5: 'Сильная пробка',
            6: 'Очень сильная пробка',
            7: 'Коллапс',
            8: 'Полный коллапс',
            9: 'Непроходимо',
            10: 'Полная остановка'
        }
        return levels.get(level, 'Неизвестно')

    def find_traffic_events(self, lat: float, lon: float, radius_m: int = 5000) -> List[Dict]:
        """
        Основной метод поиска дорожных событий
        """
        logger.info(f"Поиск пробок вокруг: {lat}, {lon}, радиус: {radius_m}м")
        
        traffic_data = self.get_traffic_data_from_html(lat, lon)
        
        if not traffic_data:
            return []
        
        # Объединяем пробки и события
        all_events = []
        
        # Добавляем пробки
        for jam in traffic_data.get('jams', []):
            if jam.get('distance_m', float('inf')) <= radius_m:
                all_events.append({
                    **jam,
                    'title': f"Пробка (уровень {jam.get('level', 0)})",
                    'severity': jam.get('level', 0)
                })
        
        # Добавляем события
        for event in traffic_data.get('events', []):
            if event.get('distance_m', float('inf')) <= radius_m:
                all_events.append({
                    **event,
                    'severity': 1  # Базовый уровень серьезности для событий
                })
        
        # Сортируем по расстоянию
        all_events.sort(key=lambda x: x.get('distance_m', float('inf')))
        
        return all_events

    def get_area_traffic_summary(self, lat: float, lon: float) -> Dict:
        """Получение сводки по пробкам в районе"""
        traffic_data = self.get_traffic_data_from_html(lat, lon)
        
        if not traffic_data:
            return {
                'overall_level': 0,
                'description': 'Данные недоступны',
                'jams_count': 0,
                'events_count': 0
            }
        
        return {
            'overall_level': traffic_data.get('overall_level', 0),
            'description': self.get_traffic_level_description(traffic_data.get('overall_level', 0)),
            'jams_count': len(traffic_data.get('jams', [])),
            'events_count': len(traffic_data.get('events', [])),
            'timestamp': traffic_data.get('timestamp')
        }


def main():
    """Демонстрация работы парсера"""
    print("🚦 Парсер дорожных событий Яндекс Карт")
    print("=" * 50)
    
    parser = YandexTrafficParser()
    
    # Тестовые локации
    locations = [
    ("Ростов-на-Дону, ул. Большая Садовая, 1 (VIP)", 47.217855, 39.696085),
    ("Ростов-на-Дону, пр. Будённовский, 15", 47.217818, 39.708419),
    ("Ростов-на-Дону, ул. Красноармейская, 67", 47.228663, 39.714995),
    ("Ростов-на-Дону, пр. Ворошиловский, 32", 47.225499, 39.717789),
    ("Ростов-на-Дону, ул. Таганрогская, 124", 47.256063, 39.644531),
    ("Ростов-на-Дону, ул. Пушкинская, 89", 47.224703, 39.711402),
    ("Ростов-на-Дону, пр. Стачки, 45", 47.211759, 39.663306),
    ("Ростов-на-Дону, ул. Малиновского, 76", 47.233968, 39.614608),
    ("Ростов-на-Дону, ул. Горького, 23", 47.223290, 39.697406),
    ("Ростов-на-Дону, ул. Социалистическая, 54", 47.218700, 39.707440)
    ]
    
    radius = 250  # радиус
    results = []
    
    for name, lat, lon in locations:
        try:
            print(f"\n📍 {name}")
            print(f"📌 Координаты: {lat}, {lon}")
            
            # Получаем сводку по пробкам
            summary = parser.get_area_traffic_summary(lat, lon)
            print(f"📊 Общий уровень пробок: {summary['overall_level']} - {summary['description']}")
            print(f"🔍 Пробок: {summary['jams_count']}, Событий: {summary['events_count']}")
            
            # Ищем события в радиусе
            events = parser.find_traffic_events(lat, lon, radius)
            
            result = {
                "location": name,
                "coordinates": [lat, lon],
                "radius_m": radius,
                "traffic_summary": summary,
                "found_events": len(events),
                "events": events,
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
            
            print(f"🎯 Найдено событий в радиусе {radius}м: {len(events)}")
            
            # Вывод событий
            for i, event in enumerate(events, 1):
                print(f"   {i}. {event.get('title', 'Событие')}")
                print(f"      📍 {event.get('distance_m', 0)}м | 🚦 Уровень: {event.get('severity', 'N/A')}")
                if event.get('description'):
                    print(f"      📝 {event['description']}")
                print()
            
            # Пауза между запросами
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Ошибка для локации {name}: {e}")
            continue
    
    # Сохранение результатов
    if results:
        filename = f"yandex_traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Результаты сохранены в {filename}")
    
    # Статистика
    total_events = sum(r['found_events'] for r in results)
    print(f"\n📊 ИТОГО: {total_events} событий на {len(results)} локациях")


if __name__ == "__main__":
    main()