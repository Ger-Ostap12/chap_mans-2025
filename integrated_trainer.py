"""
Тренер для интегрированной системы: Graph Transformer + ANN
Обучает ANN на обогащенных данных с графовыми признаками
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict
from real_distance_analyzer import RealDistanceAnalyzer # Импортируем анализатор расстояний

class IntegratedRouteDataset(Dataset):
    """
    Датасет для обучения интегрированной системы
    Создает обогащенные данные: 7 исходных + 7 графовых = 14 признаков
    """

    def __init__(self, data_dir: str = "DS", use_real_distances: bool = True,
                 tomtom_api_key: str = None):
        self.data_dir = data_dir
        self.use_real_distances = use_real_distances

        # Инициализируем анализатор расстояний
        if use_real_distances:
            self.distance_analyzer = RealDistanceAnalyzer(tomtom_api_key)
            self.distance_analyzer.load_nyc_trip_data()
            self.distance_analyzer.load_zone_geometries()

        self.trips = self._load_nyc_data()

    def _load_nyc_data(self):
        """
        Загрузка NYC данных для создания маршрутов
        """
        try:
            # Загружаем объединенные данные
            combined_data_path = "combined_training_data.csv"

            if os.path.exists(combined_data_path):
                print("📊 Загрузка объединенных данных для интегрированного обучения...")
                df = pd.read_csv(combined_data_path)

                trips = []
                for _, row in df.iterrows():
                    trip = {
                        'lat': row.get('latitude', 0.0),
                        'lon': row.get('longitude', 0.0),
                        'is_vip': row.get('is_vip', 0),
                        'work_start_hour': row.get('work_start_hour', 0.0),
                        'work_end_hour': row.get('work_end_hour', 0.0),
                        'lunch_start_hour': row.get('lunch_start_hour', 0.0),
                        'lunch_end_hour': row.get('lunch_end_hour', 0.0)
                    }
                    trips.append(trip)

                print(f"✅ Загружено {len(trips)} записей для интегрированного обучения")
                return trips
            else:
                print("❌ Файл combined_training_data.csv не найден!")
                return []

        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            return []

    def __len__(self):
        return len(self.trips)

    def __getitem__(self, idx):
        """
        Создает маршрут из NYC данных
        """
        # Берем 5-10 случайных поездок для создания маршрута
        route_size = np.random.randint(5, 11)
        start_idx = idx % (len(self.trips) - route_size)

        route_trips = self.trips[start_idx:start_idx + route_size]

        # Создаем клиентов
        clients = []
        for i, trip in enumerate(route_trips):
            client = {
                'lat': trip['lat'],
                'lon': trip['lon'],
                'client_level': 'VIP' if trip['is_vip'] == 1 else 'regular',
                'work_start_hour': trip['work_start_hour'],
                'work_end_hour': trip['work_end_hour'],
                'lunch_start_hour': trip['lunch_start_hour'],
                'lunch_end_hour': trip['lunch_end_hour']
            }
            clients.append(client)

        # Создаем оптимальный порядок (VIP первый, остальные по расстоянию)
        optimal_order = self._create_optimal_order(clients)

        return {
            'clients': clients,
            'optimal_order': optimal_order,
            'route_id': f"route_{idx}"
        }

    def _create_optimal_order(self, clients: List[Dict]) -> List[int]:
        """
        Создает оптимальный порядок посещения клиентов
        """
        num_clients = len(clients)

        # VIP клиенты получают приоритет
        vip_indices = [i for i, client in enumerate(clients) if client['client_level'] == 'VIP']
        regular_indices = [i for i, client in enumerate(clients) if client['client_level'] == 'regular']

        # Простая эвристика: VIP первый, остальные по ближайшему соседу
        optimal_order = vip_indices.copy()

        if regular_indices:
            # Добавляем обычных клиентов по ближайшему соседу
            current = vip_indices[0] if vip_indices else regular_indices[0]
            remaining = regular_indices.copy()

            while remaining:
                distances = []
                for next_idx in remaining:
                    # Используем реальные расстояния если доступны
                    if self.use_real_distances and hasattr(self, 'distance_analyzer'):
                        dist = self.distance_analyzer.get_rostov_distance(
                            clients[current]['lat'], clients[current]['lon'],
                            clients[next_idx]['lat'], clients[next_idx]['lon']
                        )
                    else:
                        dist = np.sqrt((clients[current]['lat'] - clients[next_idx]['lat'])**2 +
                                     (clients[current]['lon'] - clients[next_idx]['lon'])**2)
                    distances.append((dist, next_idx))

                distances.sort()
                next_client = distances[0][1]
                optimal_order.append(next_client)
                remaining.remove(next_client)
                current = next_client

        return optimal_order

class IntegratedTrainer:
    """
    Тренер для интегрированной системы
    """

    def __init__(self, ann_model, graph_extractor, device='cpu'):
        self.ann_model = ann_model
        self.graph_extractor = graph_extractor
        self.device = device

        # Переносим модели на устройство
        self.ann_model.to(device)
        self.graph_extractor.to(device)

    def train_epoch(self, dataloader, optimizer, criterion):
        """
        Обучение одной эпохи интегрированной системы
        """
        self.ann_model.train()
        self.graph_extractor.train()

        total_loss = 0

        for batch in dataloader:
            clients = batch['clients']
            optimal_order = batch['optimal_order']

            # Этап 1: Graph Transformer извлекает графовые признаки
            enriched_features = self.graph_extractor.extract_graph_features(clients)

            # Этап 2: ANN обрабатывает обогащенные признаки
            features_tensor = torch.tensor(enriched_features, dtype=torch.float).to(self.device)

            # ANN предсказание
            ann_output = self.ann_model(features_tensor)

            # Создаем целевые значения
            target_scores = torch.zeros(len(clients))
            for i, order_idx in enumerate(optimal_order):
                target_scores[order_idx] = len(clients) - i  # Высший приоритет = большее число

            target_scores = target_scores.to(self.device)

            # Вычисляем loss
            loss = criterion(ann_output.squeeze(), target_scores)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, train_data: List[Dict], epochs: int = 100,
              batch_size: int = 4, learning_rate: float = 0.001,
              save_path: str = 'integrated_model.pth'):
        """
        Полное обучение интегрированной системы
        """
        print(f"🧠 Начинаем обучение интегрированной системы...")
        print(f"📊 Данных для обучения: {len(train_data)}")
        print(f"🎯 Эпох: {epochs}")
        print(f"📦 Batch size: {batch_size}")
        print(f"📈 Learning rate: {learning_rate}")

        # Создаем датасет и даталоадер
        dataset = IntegratedRouteDataset()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Оптимизатор и loss функция
        optimizer = optim.Adam(self.ann_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # История обучения
        train_losses = []

        print(f"\n🚀 Начинаем обучение интегрированной системы...")

        for epoch in range(epochs):
            # Обучение
            train_loss = self.train_epoch(dataloader, optimizer, criterion)
            train_losses.append(train_loss)

            # Выводим прогресс
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Эпоха {epoch+1}/{epochs}, Loss: {train_loss:.6f}")

        # Сохраняем модель
        torch.save(self.ann_model.state_dict(), save_path)
        print(f"\n✅ Интегрированное обучение завершено!")
        print(f"💾 ANN модель сохранена: {save_path}")
        print(f"📉 Финальный loss: {train_losses[-1]:.6f}")

        return {
            'status': 'success',
            'epochs_trained': epochs,
            'final_loss': train_losses[-1],
            'train_losses': train_losses,
            'model_saved': save_path
        }

def main():
    """
    Основная функция обучения интегрированной системы
    """
    print("🧠 Integrated System Trainer")
    print("=" * 50)

    # Импортируем модели
    from tomtom_traffic_router import GraphTransformerFeatureExtractor
    from train_model import AttentionRouteOptimizer

    # Создаем модели
    graph_extractor = GraphTransformerFeatureExtractor(
        input_dim=7,
        hidden_dim=128,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    )

    ann_model = AttentionRouteOptimizer(
        input_dim=14,  # 7 исходных + 7 графовых
        hidden_dim=256,
        num_heads=8,
        num_layers=3
    )

    # Создаем тренер
    trainer = IntegratedTrainer(ann_model, graph_extractor)

    # Создаем данные для обучения с реальными расстояниями
    dataset = IntegratedRouteDataset(
        use_real_distances=True,
        tomtom_api_key="zq6uGYYW806zBrLssIilztKU6ixjAOkr"
    )
    training_data = [dataset[i] for i in range(min(100, len(dataset)))]  # 100 примеров

    if not training_data:
        print("❌ Нет данных для обучения!")
        return

    # Обучаем интегрированную систему
    result = trainer.train(
        train_data=training_data,
        epochs=50,
        batch_size=2,
        learning_rate=0.001,
        save_path='integrated_model.pth'
    )

    if result['status'] == 'success':
        print(f"\n🎉 Интегрированное обучение успешно завершено!")
        print(f"📊 Эпох обучено: {result['epochs_trained']}")
        print(f"📉 Финальный loss: {result['final_loss']:.6f}")
        print(f"💾 Модель сохранена: {result['model_saved']}")
        print(f"\n🚀 Теперь можно использовать интегрированную систему!")
    else:
        print(f"❌ Ошибка обучения: {result.get('message', 'Неизвестная ошибка')}")

if __name__ == "__main__":
    main()
