// 🗺️ Пример использования API для динамического изменения местоположения
// Frontend JavaScript код

class RouteOptimizer {
    constructor(apiBaseUrl = 'http://localhost:8000') {
        this.apiBaseUrl = apiBaseUrl;
        this.currentLocation = null;
        this.currentRoutes = null;
    }

    // 1. Определение местоположения пользователя
    async detectUserLocation() {
        console.log('📍 Определение местоположения пользователя...');

        // Способ 1: GPS координаты
        if (navigator.geolocation) {
            try {
                const position = await this.getCurrentPosition();
                const gpsCoords = [position.coords.latitude, position.coords.longitude];

                const response = await fetch(`${this.apiBaseUrl}/set_location`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ gps_coords: gpsCoords })
                });

                const result = await response.json();
                if (result.success) {
                    this.currentLocation = result.location;
                    console.log('✅ GPS местоположение установлено:', result.location.address);
                    return result;
                }
            } catch (error) {
                console.log('❌ GPS недоступен, пробуем IP...');
            }
        }

        // Способ 2: IP геолокация (автоматически)
        try {
            const response = await fetch(`${this.apiBaseUrl}/set_location`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });

            const result = await response.json();
            if (result.success) {
                this.currentLocation = result.location;
                console.log('✅ IP местоположение установлено:', result.location.address);
                return result;
            }
        } catch (error) {
            console.error('❌ Ошибка определения местоположения:', error);
        }
    }

    // 2. Обновление местоположения (пользователь перетащил маркер на карте)
    async updateLocationFromMap(latitude, longitude) {
        console.log(`🗺️ Обновление местоположения: ${latitude}, ${longitude}`);

        try {
            const response = await fetch(`${this.apiBaseUrl}/update_location`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ latitude, longitude })
            });

            const result = await response.json();
            if (result.success) {
                this.currentLocation = result.new_location;
                console.log('✅ Местоположение обновлено:', result.new_location.address);

                // Автоматически пересчитываем маршруты
                if (this.currentRoutes) {
                    await this.recalculateRoutes();
                }

                return result;
            }
        } catch (error) {
            console.error('❌ Ошибка обновления местоположения:', error);
        }
    }

    // 3. Поиск предложений адресов (автодополнение)
    async searchAddressSuggestions(query) {
        console.log(`🔍 Поиск предложений для: ${query}`);

        try {
            const response = await fetch(`${this.apiBaseUrl}/location_suggestions?q=${encodeURIComponent(query)}&limit=5`);
            const result = await response.json();

            if (result.success) {
                console.log(`✅ Найдено ${result.suggestions.length} предложений`);
                return result.suggestions;
            }
        } catch (error) {
            console.error('❌ Ошибка поиска предложений:', error);
        }
        return [];
    }

    // 4. Установка местоположения по адресу
    async setLocationByAddress(address) {
        console.log(`🏠 Установка местоположения по адресу: ${address}`);

        try {
            const response = await fetch(`${this.apiBaseUrl}/set_location`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ manual_address: address })
            });

            const result = await response.json();
            if (result.success) {
                this.currentLocation = result.location;
                console.log('✅ Адресное местоположение установлено:', result.location.address);
                return result;
            }
        } catch (error) {
            console.error('❌ Ошибка установки адреса:', error);
        }
    }

    // 5. Оптимизация маршрутов от местоположения пользователя
    async optimizeRoutesFromLocation(clients, numDays = 3) {
        console.log(`🚗 Оптимизация маршрутов для ${clients.length} клиентов на ${numDays} дней...`);

        try {
            const response = await fetch(`${this.apiBaseUrl}/optimize_routes_from_location`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ clients_data: clients, num_days: numDays })
            });

            const result = await response.json();
            if (result.success) {
                this.currentRoutes = result;
                console.log('✅ Маршруты оптимизированы:', result);
                return result;
            }
        } catch (error) {
            console.error('❌ Ошибка оптимизации маршрутов:', error);
        }
    }

    // 6. Пересчет маршрутов с нового местоположения
    async recalculateRoutes() {
        if (!this.currentRoutes) {
            console.log('⚠️ Нет текущих маршрутов для пересчета');
            return;
        }

        console.log('🔄 Пересчет маршрутов с нового местоположения...');

        try {
            const clients = this.currentRoutes.clients || [];
            const numDays = this.currentRoutes.num_days || 3;

            const response = await fetch(`${this.apiBaseUrl}/recalculate_routes`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ clients_data: clients, num_days: numDays })
            });

            const result = await response.json();
            if (result.success) {
                this.currentRoutes = result.routes;
                console.log('✅ Маршруты пересчитаны:', result);
                return result;
            }
        } catch (error) {
            console.error('❌ Ошибка пересчета маршрутов:', error);
        }
    }

    // 7. Отметка клиента как посещенного
    async markClientVisited(clientId, actualServiceTime = null) {
        console.log(`✅ Отметка клиента ${clientId} как посещенного...`);

        try {
            const response = await fetch(`${this.apiBaseUrl}/mark_visited`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    client_id: clientId,
                    actual_service_time: actualServiceTime
                })
            });

            const result = await response.json();
            if (result.success) {
                console.log('✅ Клиент отмечен как посещенный:', result);
                return result;
            }
        } catch (error) {
            console.error('❌ Ошибка отметки клиента:', error);
        }
    }

    // 8. Получение текущего местоположения
    async getCurrentLocation() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/current_location`);
            const result = await response.json();

            if (result.success) {
                this.currentLocation = result.location;
                return result.location;
            }
        } catch (error) {
            console.error('❌ Ошибка получения местоположения:', error);
        }
        return null;
    }

    // Вспомогательные методы
    getCurrentPosition() {
        return new Promise((resolve, reject) => {
            navigator.geolocation.getCurrentPosition(resolve, reject, {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 300000
            });
        });
    }

    // Получение координат из маршрута для отображения на карте
    getRouteWaypoints(routeData) {
        if (!routeData || !routeData.routes) return [];

        const waypoints = [];
        routeData.routes.forEach(dayRoute => {
            if (dayRoute.clients) {
                dayRoute.clients.forEach(client => {
                    waypoints.push({
                        lat: client.lat,
                        lon: client.lon,
                        address: client.address,
                        client_id: client.id,
                        is_vip: client.client_level === 'VIP'
                    });
                });
            }
        });

        return waypoints;
    }
}

// Пример использования
async function example() {
    const optimizer = new RouteOptimizer();

    // 1. Определяем местоположение пользователя
    await optimizer.detectUserLocation();

    // 2. Загружаем клиентов (пример данных)
    const clients = [
        {
            id: 1,
            address: "Ростов-на-Дону, ул. Большая Садовая, 1",
            lat: 47.2225,
            lon: 39.7203,
            client_level: "VIP",
            work_start: "09:00",
            work_end: "18:00",
            lunch_start: "13:00",
            lunch_end: "14:00"
        },
        {
            id: 2,
            address: "Ростов-на-Дону, ул. Театральная, 5",
            lat: 47.2300,
            lon: 39.7300,
            client_level: "Стандарт",
            work_start: "09:00",
            work_end: "18:00",
            lunch_start: "13:00",
            lunch_end: "14:00"
        }
    ];

    // 3. Оптимизируем маршруты
    const routes = await optimizer.optimizeRoutesFromLocation(clients, 2);

    // 4. Получаем точки для карты
    const waypoints = optimizer.getRouteWaypoints(routes);
    console.log('📍 Точки маршрута:', waypoints);

    // 5. Симуляция изменения местоположения пользователем
    setTimeout(async () => {
        await optimizer.updateLocationFromMap(47.2500, 39.7500);
    }, 5000);

    // 6. Симуляция отметки клиента как посещенного
    setTimeout(async () => {
        await optimizer.markClientVisited(1, 25); // 25 минут вместо 30
    }, 10000);
}

// Запуск примера
// example();

export default RouteOptimizer;
