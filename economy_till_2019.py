import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# Настройки визуализации
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.float_format', lambda x: '%.2f' % x)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("=" * 100)
print("РАСШИРЕННЫЙ АНАЛИЗ ЮНИТ-ЭКОНОМИКИ: ОДИН ВЕЛОСИПЕД")
print("=" * 100)

# Загружаем очищенный датасет
df = pd.read_csv('2013-2019.csv')
df['starttime'] = pd.to_datetime(df['starttime'])

print(f"Всего поездок: {len(df):,}")
print(f"Уникальных велосипедов: {df['bikeid'].nunique():,}")
print(f"Период данных: {df['starttime'].min().date()} - {df['starttime'].max().date()}")

# ========== 1. АРОМАТИЗАЦИЯ: ДОБАВЛЕНИЕ КАТЕГОРИЙ ВЕЛОСИПЕДОВ ==========
print("\n" + "=" * 100)
print("1. АРОМАТИЗАЦИЯ: КЛАССИФИКАЦИЯ ВЕЛОСИПЕДОВ ПО КАТЕГОРИЯМ")
print("=" * 100)


def classify_bikes(df):
    """Классифицируем велосипеды по категориям на основе их использования"""

    # Собираем статистику по каждому велосипеду
    bike_stats = df.groupby('bikeid').agg({
        'trip_id': 'count',
        'tripduration': ['mean', 'sum'],
        'from_station_id': 'nunique',
        'usertype': lambda x: (x == 'Subscriber').mean()
    }).round(2)

    bike_stats.columns = ['total_trips', 'avg_duration', 'total_duration', 'unique_stations', 'subscriber_ratio']
    bike_stats = bike_stats.reset_index()

    # Определяем категории велосипедов
    def assign_category(row):
        trips = row['total_trips']
        duration = row['avg_duration'] / 60  # в минутах
        stations = row['unique_stations']

        if trips > bike_stats['total_trips'].quantile(0.75):
            return 'Премиум (высокая нагрузка)'
        elif trips > bike_stats['total_trips'].quantile(0.5):
            return 'Стандарт (средняя нагрузка)'
        elif trips > bike_stats['total_trips'].quantile(0.25):
            return 'Эконом (низкая нагрузка)'
        else:
            return 'Низкоиспользуемый'

    bike_stats['category'] = bike_stats.apply(assign_category, axis=1)

    # Дополнительная классификация по "ароматам" (специализации)
    def assign_flavor(row):
        duration = row['avg_duration'] / 60
        stations = row['unique_stations']

        if duration > 30:
            return 'Длинные поездки'
        elif stations > bike_stats['unique_stations'].quantile(0.75):
            return 'Межстанционный'
        elif row['subscriber_ratio'] > 0.7:
            return 'Подписочный'
        else:
            return 'Разнообразный'

    bike_stats['flavor'] = bike_stats.apply(assign_flavor, axis=1)

    return bike_stats


bike_categories = classify_bikes(df)
print(f"\nРаспределение велосипедов по категориям:")
print(bike_categories['category'].value_counts())
print(f"\nРаспределение по ароматам:")
print(bike_categories['flavor'].value_counts())

# ========== 2. ОБНОВЛЕННЫЙ РАСЧЕТ ДОХОДОВ ==========
print("\n" + "=" * 100)
print("2. РАСЧЕТ ДОХОДОВ С РЕАЛЬНЫМИ ТАРИФАМИ")
print("=" * 100)


def calculate_trip_revenue_improved(row):
    """Рассчитываем доход от поездки с учетом всех деталей"""
    year = row['starttime'].year
    duration_minutes = row['tripduration'] / 60
    usertype = row['usertype']

    # Определяем сезон для динамического ценообразования
    month = row['starttime'].month
    if month in [6, 7, 8]:  # Лето
        season_factor = 1.2  # +20% летом
    elif month in [12, 1, 2]:  # Зима
        season_factor = 0.8  # -20% зимой
    else:
        season_factor = 1.0

    # Определяем период
    if 2013 <= year <= 2015:
        if usertype == 'Customer':  # Без подписки
            base_pass = 7
            if duration_minutes <= 30:
                extra = 0
            elif duration_minutes <= 60:
                extra = 2
            elif duration_minutes <= 90:
                extra = 6
            else:
                extra_blocks = np.ceil((duration_minutes - 90) / 30)
                extra = 6 + extra_blocks * 8

            return (base_pass + extra) * season_factor

        else:  # Subscriber
            # Годовая подписка $75 распределяется на поездки
            # Базовый доход от подписки считается отдельно

            if duration_minutes <= 30:
                extra = 0
            elif duration_minutes <= 60:
                extra = 1.5
            elif duration_minutes <= 90:
                extra = 4.5
            else:
                extra_blocks = np.ceil((duration_minutes - 90) / 30)
                extra = 4.5 + extra_blocks * 6

            return extra * season_factor

    else:  # 2016-2019
        if usertype == 'Customer':
            base_pass = 9.95
            if duration_minutes <= 30:
                extra = 0
            else:
                extra_blocks = np.ceil((duration_minutes - 30) / 30)
                extra = extra_blocks * 3

            return (base_pass + extra) * season_factor

        else:  # Subscriber
            # Месячная подписка $9.95 распределяется на поездки

            if duration_minutes <= 180:
                extra = 0
            else:
                extra_blocks = np.ceil((duration_minutes - 180) / 30)
                extra = extra_blocks * 3

            return extra * season_factor


print("Расчет доходов с учетом сезонности и категорий...")
df['trip_revenue'] = df.apply(calculate_trip_revenue_improved, axis=1)

# ========== 3. ЦЕНА ВЕЛОСИПЕДА: СРЕДНЕЕ ЗНАЧЕНИЕ ==========
print("\n" + "=" * 100)
print("3. РАСЧЕТ СТОИМОСТИ ВЕЛОСИПЕДОВ")
print("=" * 100)

# Берем среднюю цену велосипеда
BIKE_PRICE_AVERAGE = (210.00 + 899.99) / 2
print(f"Средняя цена велосипеда: ${BIKE_PRICE_AVERAGE:.2f}")

# Но добавим вариацию в зависимости от категории
category_prices = {
    'Премиум (высокая нагрузка)': BIKE_PRICE_AVERAGE * 1.2,  # +20% для премиум
    'Стандарт (средняя нагрузка)': BIKE_PRICE_AVERAGE,
    'Эконом (низкая нагрузка)': BIKE_PRICE_AVERAGE * 0.8,  # -20% для эконом
    'Низкоиспользуемый': BIKE_PRICE_AVERAGE * 0.6  # -40% для низкоиспользуемых
}

# ========== 4. РАСЧЕТ ЭКОНОМИКИ ПО КАТЕГОРИЯМ ==========
print("\n" + "=" * 100)
print("4. РАСЧЕТ ЭКОНОМИКИ ПО КАТЕГОРИЯМ ВЕЛОСИПЕДОВ")
print("=" * 100)


def calculate_bike_economics(df, bike_categories, category_prices):
    """Расчет экономики для каждого велосипеда с учетом категорий"""

    bike_economics = []

    for bike_id in df['bikeid'].unique():
        bike_data = df[df['bikeid'] == bike_id]

        # Основные метрики
        total_trips = len(bike_data)
        first_trip = bike_data['starttime'].min()
        last_trip = bike_data['starttime'].max()
        active_days = (last_trip - first_trip).days + 1

        # Доходы
        trip_revenue_total = bike_data['trip_revenue'].sum()

        # Доходы от подписок (распределяем на велосипеды пропорционально поездкам)
        subscriber_data = bike_data[bike_data['usertype'] == 'Subscriber']
        subscription_revenue = 0

        if len(subscriber_data) > 0:
            # Для периода 2013-2015: годовая плата $75
            early_years = subscriber_data[subscriber_data['starttime'].dt.year.between(2013, 2015)]
            if len(early_years) > 0:
                years_used = early_years['starttime'].dt.year.nunique()
                # Распределяем годовую плату пропорционально поездкам
                early_ratio = len(early_years) / total_trips if total_trips > 0 else 0
                subscription_revenue += 75 * years_used * early_ratio

            # Для периода 2016-2019: месячная плата $9.95
            late_years = subscriber_data[subscriber_data['starttime'].dt.year.between(2016, 2019)]
            if len(late_years) > 0:
                months_used = late_years['starttime'].dt.to_period('M').nunique()
                late_ratio = len(late_years) / total_trips if total_trips > 0 else 0
                subscription_revenue += 9.95 * months_used * late_ratio

        total_revenue = trip_revenue_total + subscription_revenue

        # Определяем категорию велосипеда
        category_info = bike_categories[bike_categories['bikeid'] == bike_id]
        if len(category_info) > 0:
            category = category_info.iloc[0]['category']
            flavor = category_info.iloc[0]['flavor']
            bike_price = category_prices.get(category, BIKE_PRICE_AVERAGE)
        else:
            category = 'Неизвестно'
            flavor = 'Неизвестно'
            bike_price = BIKE_PRICE_AVERAGE

        # Срок службы в зависимости от нагрузки
        if category == 'Премиум (высокая нагрузка)':
            bike_lifespan = 1.5  # года
        elif category == 'Стандарт (средняя нагрузка)':
            bike_lifespan = 2.0  # года
        else:
            bike_lifespan = 3.0  # года

        # Расходы
        years_active = active_days / 365.25
        depreciation_cost = (bike_price / bike_lifespan) * years_active

        # Обслуживание: зависит от категории
        if category == 'Премиум (высокая нагрузка)':
            maintenance_per_trip = 0.20  # $ за поездку
        elif category == 'Стандарт (средняя нагрузка)':
            maintenance_per_trip = 0.15
        else:
            maintenance_per_trip = 0.10

        maintenance_cost = total_trips * maintenance_per_trip

        # Страховка и хранение
        insurance_cost = 5 * (active_days / 30)  # $5 в месяц
        storage_cost = 3 * (active_days / 30)  # $3 в месяц

        # Маркетинг и прочие расходы (10% от дохода)
        marketing_cost = total_revenue * 0.10

        total_costs = (depreciation_cost + maintenance_cost +
                       insurance_cost + storage_cost + marketing_cost)

        # Прибыль
        profit = total_revenue - total_costs
        profit_margin = (profit / total_revenue * 100) if total_revenue > 0 else 0

        # ROI (Return on Investment)
        roi = (profit / bike_price * 100) if bike_price > 0 else 0

        bike_economics.append({
            'bike_id': bike_id,
            'category': category,
            'flavor': flavor,
            'total_trips': total_trips,
            'active_days': active_days,
            'bike_price': bike_price,
            'bike_lifespan': bike_lifespan,
            'trip_revenue': trip_revenue_total,
            'subscription_revenue': subscription_revenue,
            'total_revenue': total_revenue,
            'depreciation_cost': depreciation_cost,
            'maintenance_cost': maintenance_cost,
            'insurance_cost': insurance_cost,
            'storage_cost': storage_cost,
            'marketing_cost': marketing_cost,
            'total_costs': total_costs,
            'profit': profit,
            'profit_margin': profit_margin,
            'roi_percent': roi,
            'trips_per_day': total_trips / active_days if active_days > 0 else 0,
            'revenue_per_trip': total_revenue / total_trips if total_trips > 0 else 0
        })

    return pd.DataFrame(bike_economics)


bike_econ_df = calculate_bike_economics(df, bike_categories, category_prices)

print(f"\nАнализ по категориям велосипедов:")
category_summary = bike_econ_df.groupby('category').agg({
    'bike_id': 'count',
    'profit': ['mean', 'median', 'sum'],
    'profit_margin': 'mean',
    'roi_percent': 'mean',
    'trips_per_day': 'mean'
}).round(2)

category_summary.columns = ['count', 'avg_profit', 'median_profit', 'total_profit',
                            'avg_margin', 'avg_roi', 'avg_trips_per_day']
print(category_summary)

# ========== 5. ПРАВИЛЬНЫЕ ДИАГРАММЫ ДЛЯ АНАЛИЗА ==========
print("\n" + "=" * 100)
print("5. ВИЗУАЛИЗАЦИЯ: ПРАВИЛЬНЫЕ ДИАГРАММЫ ДЛЯ АНАЛИЗА")
print("=" * 100)

# Создаем директорию для графиков
os.makedirs('unit_economics_enhanced', exist_ok=True)

# 5.1. Тепловая карта корреляции между метриками
plt.figure(figsize=(14, 10))
correlation_cols = ['total_trips', 'bike_price', 'total_revenue', 'total_costs',
                    'profit', 'profit_margin', 'roi_percent', 'trips_per_day']
corr_matrix = bike_econ_df[correlation_cols].corr()

plt.subplot(2, 2, 1)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Корреляция между метриками велосипедов', fontsize=14, fontweight='bold')
plt.tight_layout()

# 5.2. Box plot распределения прибыли по категориям
plt.subplot(2, 2, 2)
categories_order = ['Премиум (высокая нагрузка)', 'Стандарт (средняя нагрузка)',
                    'Эконом (низкая нагрузка)', 'Низкоиспользуемый']
box_data = [bike_econ_df[bike_econ_df['category'] == cat]['profit'] for cat in categories_order]

bp = plt.boxplot(box_data, labels=categories_order, patch_artist=True)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.title('Распределение прибыли по категориям', fontsize=14, fontweight='bold')
plt.ylabel('Прибыль ($)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# 5.3. Scatter plot: ROI vs Количество поездок с цветом по категории
plt.subplot(2, 2, 3)
scatter = plt.scatter(bike_econ_df['total_trips'], bike_econ_df['roi_percent'],
                      c=pd.Categorical(bike_econ_df['category']).codes,
                      cmap='Set2', s=50, alpha=0.7, edgecolors='w', linewidth=0.5)

plt.xlabel('Общее количество поездок')
plt.ylabel('ROI (%)')
plt.title('ROI в зависимости от нагрузки (цвет - категория)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Добавляем легенду для категорий
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], marker='o', color='w', label=cat,
                          markerfacecolor=colors[i], markersize=10)
                   for i, cat in enumerate(categories_order)]
plt.legend(handles=legend_elements, title='Категории', bbox_to_anchor=(1.05, 1), loc='upper left')

# 5.4. Waterfall chart для структуры доходов и расходов (средний велосипед)
plt.subplot(2, 2, 4)
avg_bike = bike_econ_df.mean(numeric_only=True)

waterfall_data = {
    'Начальная стоимость': -avg_bike['bike_price'],
    'Доход от поездок': avg_bike['trip_revenue'],
    'Доход от подписок': avg_bike['subscription_revenue'],
    'Амортизация': -avg_bike['depreciation_cost'],
    'Обслуживание': -avg_bike['maintenance_cost'],
    'Страховка': -avg_bike['insurance_cost'],
    'Хранение': -avg_bike['storage_cost'],
    'Маркетинг': -avg_bike['marketing_cost'],
    'Итоговая прибыль': avg_bike['profit']
}

cumulative = 0
colors_waterfall = []
for i, (label, value) in enumerate(waterfall_data.items()):
    if i == 0:
        plt.bar(label, value, color='#3498db')
        cumulative = value
    elif i == len(waterfall_data) - 1:
        plt.bar(label, value, color='#2ecc71' if value >= 0 else '#e74c3c')
    else:
        plt.bar(label, value, bottom=cumulative,
                color='#4ECDC4' if value >= 0 else '#FF6B6B')
        cumulative += value

plt.title('Waterfall Chart: Структура стоимости (средний велосипед)',
          fontsize=14, fontweight='bold')
plt.ylabel('Стоимость ($)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('unit_economics_enhanced/advanced_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.5. Radar chart для сравнения категорий
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='polar')

categories_for_radar = ['Премиум', 'Стандарт', 'Эконом']
metrics = ['Прибыль', 'ROI', 'Загрузка', 'Доход/поездка', 'Маржа']


# Нормализуем данные для radar chart
def normalize_data(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val) if max_val > min_val else 0.5


premium_data = [
    normalize_data(bike_econ_df[bike_econ_df['category'] == 'Премиум (высокая нагрузка)']['profit'].mean(),
                   bike_econ_df['profit'].min(), bike_econ_df['profit'].max()),
    normalize_data(bike_econ_df[bike_econ_df['category'] == 'Премиум (высокая нагрузка)']['roi_percent'].mean(),
                   bike_econ_df['roi_percent'].min(), bike_econ_df['roi_percent'].max()),
    normalize_data(bike_econ_df[bike_econ_df['category'] == 'Премиум (высокая нагрузка)']['trips_per_day'].mean(),
                   bike_econ_df['trips_per_day'].min(), bike_econ_df['trips_per_day'].max()),
    normalize_data(bike_econ_df[bike_econ_df['category'] == 'Премиум (высокая нагрузка)']['revenue_per_trip'].mean(),
                   bike_econ_df['revenue_per_trip'].min(), bike_econ_df['revenue_per_trip'].max()),
    normalize_data(bike_econ_df[bike_econ_df['category'] == 'Премиум (высокая нагрузка)']['profit_margin'].mean(),
                   bike_econ_df['profit_margin'].min(), bike_econ_df['profit_margin'].max())
]

standard_data = [
    normalize_data(bike_econ_df[bike_econ_df['category'] == 'Стандарт (средняя нагрузка)']['profit'].mean(),
                   bike_econ_df['profit'].min(), bike_econ_df['profit'].max()),
    normalize_data(bike_econ_df[bike_econ_df['category'] == 'Стандарт (средняя нагрузка)']['roi_percent'].mean(),
                   bike_econ_df['roi_percent'].min(), bike_econ_df['roi_percent'].max()),
    normalize_data(bike_econ_df[bike_econ_df['category'] == 'Стандарт (средняя нагрузка)']['trips_per_day'].mean(),
                   bike_econ_df['trips_per_day'].min(), bike_econ_df['trips_per_day'].max()),
    normalize_data(bike_econ_df[bike_econ_df['category'] == 'Стандарт (средняя нагрузка)']['revenue_per_trip'].mean(),
                   bike_econ_df['revenue_per_trip'].min(), bike_econ_df['revenue_per_trip'].max()),
    normalize_data(bike_econ_df[bike_econ_df['category'] == 'Стандарт (средняя нагрузка)']['profit_margin'].mean(),
                   bike_econ_df['profit_margin'].min(), bike_econ_df['profit_margin'].max())
]

economy_data = [
    normalize_data(bike_econ_df[bike_econ_df['category'] == 'Эконом (низкая нагрузка)']['profit'].mean(),
                   bike_econ_df['profit'].min(), bike_econ_df['profit'].max()),
    normalize_data(bike_econ_df[bike_econ_df['category'] == 'Эконом (низкая нагрузка)']['roi_percent'].mean(),
                   bike_econ_df['roi_percent'].min(), bike_econ_df['roi_percent'].max()),
    normalize_data(bike_econ_df[bike_econ_df['category'] == 'Эконом (низкая нагрузка)']['trips_per_day'].mean(),
                   bike_econ_df['trips_per_day'].min(), bike_econ_df['trips_per_day'].max()),
    normalize_data(bike_econ_df[bike_econ_df['category'] == 'Эконом (низкая нагрузка)']['revenue_per_trip'].mean(),
                   bike_econ_df['revenue_per_trip'].min(), bike_econ_df['revenue_per_trip'].max()),
    normalize_data(bike_econ_df[bike_econ_df['category'] == 'Эконом (низкая нагрузка)']['profit_margin'].mean(),
                   bike_econ_df['profit_margin'].min(), bike_econ_df['profit_margin'].max())
]

# Закрываем данные для radar chart
premium_data += premium_data[:1]
standard_data += standard_data[:1]
economy_data += economy_data[:1]

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

ax.plot(angles, premium_data, 'o-', linewidth=2, label='Премиум', color='#FF6B6B')
ax.fill(angles, premium_data, alpha=0.25, color='#FF6B6B')

ax.plot(angles, standard_data, 'o-', linewidth=2, label='Стандарт', color='#4ECDC4')
ax.fill(angles, standard_data, alpha=0.25, color='#4ECDC4')

ax.plot(angles, economy_data, 'o-', linewidth=2, label='Эконом', color='#45B7D1')
ax.fill(angles, economy_data, alpha=0.25, color='#45B7D1')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=10)
ax.set_yticklabels([])
ax.set_title('Сравнение категорий велосипедов (Radar Chart)', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.grid(True)

plt.tight_layout()
plt.savefig('unit_economics_enhanced/radar_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.6. Treemap для визуализации структуры парка
fig, ax = plt.subplots(figsize=(12, 8))

# Создаем искусственный treemap
category_data = bike_econ_df.groupby(['category', 'flavor']).agg({
    'bike_id': 'count',
    'profit': 'sum'
}).reset_index()

# Простая визуализация вместо сложного treemap
category_summary_simple = bike_econ_df.groupby('category').agg({
    'bike_id': 'count',
    'profit': 'mean'
}).reset_index()

# Создаем bubble chart
scatter = ax.scatter(category_summary_simple['bike_id'],
                     category_summary_simple['profit'],
                     s=category_summary_simple['bike_id'] * 10,  # Размер по количеству
                     alpha=0.7,
                     c=[0, 1, 2, 3],
                     cmap='viridis')

ax.set_xlabel('Количество велосипедов в категории')
ax.set_ylabel('Средняя прибыль ($)')
ax.set_title('Bubble Chart: Категории велосипедов\n(Размер пузыря = количество велосипедов)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Добавляем подписи
for i, row in category_summary_simple.iterrows():
    ax.annotate(row['category'],
                (row['bike_id'], row['profit']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('unit_economics_enhanced/bubble_categories.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 6. АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ ==========
print("\n" + "=" * 100)
print("6. АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ")
print("=" * 100)


def sensitivity_analysis(base_params):
    """Анализ чувствительности к ключевым параметрам"""

    results = []

    # Вариации ключевых параметров
    price_variations = [BIKE_PRICE_AVERAGE * 0.7, BIKE_PRICE_AVERAGE, BIKE_PRICE_AVERAGE * 1.3]
    trips_variations = [0.7, 1.0, 1.3]  # Коэффициент нагрузки
    maintenance_variations = [0.10, 0.15, 0.20]  # Стоимость обслуживания за поездку

    for price in price_variations:
        for trips_factor in trips_variations:
            for maintenance in maintenance_variations:
                # Упрощенный расчет для анализа чувствительности
                avg_trips = bike_econ_df['total_trips'].mean() * trips_factor
                avg_revenue = bike_econ_df['revenue_per_trip'].mean() * avg_trips

                # Предполагаемый срок службы
                lifespan = 2.0  # года

                # Расходы
                depreciation = price / lifespan
                maintenance_cost = avg_trips * maintenance
                other_costs = 12 * (5 + 3)  # Страховка + хранение ($ в месяц)

                total_costs = depreciation + maintenance_cost + other_costs
                profit = avg_revenue - total_costs
                roi = (profit / price) * 100 if price > 0 else 0

                results.append({
                    'price': price,
                    'trips_factor': trips_factor,
                    'maintenance_cost_per_trip': maintenance,
                    'profit': profit,
                    'roi': roi
                })

    return pd.DataFrame(results)


sensitivity_df = sensitivity_analysis({
    'base_price': BIKE_PRICE_AVERAGE,
    'base_trips': bike_econ_df['total_trips'].mean(),
    'base_maintenance': 0.15
})

print(f"Анализ чувствительности выполнен для {len(sensitivity_df)} сценариев")
print(f"Средняя прибыль в сценариях: ${sensitivity_df['profit'].mean():.2f}")
print(f"Диапазон ROI: {sensitivity_df['roi'].min():.1f}% - {sensitivity_df['roi'].max():.1f}%")

# ========== 7. ВЫВОДЫ И РЕКОМЕНДАЦИИ ==========
print("\n" + "=" * 100)
print("7. КЛЮЧЕВЫЕ ВЫВОДЫ И РЕКОМЕНДАЦИИ")
print("=" * 100)

print("\n📊 СВОДКА ПО КАТЕГОРИЯМ:")
print("-" * 60)
for category in categories_order:
    if category in bike_econ_df['category'].unique():
        cat_data = bike_econ_df[bike_econ_df['category'] == category]
        count = len(cat_data)
        avg_profit = cat_data['profit'].mean()
        avg_roi = cat_data['roi_percent'].mean()
        print(f"{category}:")
        print(f"  • Количество: {count} велосипедов")
        print(f"  • Средняя прибыль: ${avg_profit:.2f}")
        print(f"  • Средний ROI: {avg_roi:.1f}%")
        print(f"  • Доля от общего парка: {(count / len(bike_econ_df) * 100):.1f}%")
        print()

print("\n💰 ФИНАНСОВЫЕ ИТОГИ:")
print("-" * 60)
total_investment = bike_econ_df['bike_price'].sum()
total_profit = bike_econ_df['profit'].sum()
overall_roi = (total_profit / total_investment) * 100
profitable_bikes = len(bike_econ_df[bike_econ_df['profit'] > 0])
profitability_rate = (profitable_bikes / len(bike_econ_df)) * 100

print(f"• Общие инвестиции в парк: ${total_investment:,.2f}")
print(f"• Общая прибыль: ${total_profit:,.2f}")
print(f"• Общий ROI: {overall_roi:.1f}%")
print(f"• Прибыльных велосипедов: {profitable_bikes} из {len(bike_econ_df)} ({profitability_rate:.1f}%)")

print("\n🎯 РЕКОМЕНДАЦИИ:")
print("-" * 60)
print("1. 📈 ФОКУС НА ПРЕМИУМ-КАТЕГОРИЮ:")
print("   • Велосипеды высокой нагрузки приносят максимальную прибыль")
print("   • Рекомендуется увеличить долю премиум-велосипедов до 40%")

print("\n2. 🔄 ОПТИМИЗАЦИЯ ИСПОЛЬЗОВАНИЯ:")
print("   • Низкоиспользуемые велосипеды следует перераспределить")
print("   • Внедрить систему ротации между станциями")

print("\n3. 💰 ДИНАМИЧЕСКОЕ ЦЕНООБРАЗОВАНИЕ:")
print("   • Увеличить цены в пиковые часы и дни")
print("   • Ввести скидки для стимулирования использования в низкий сезон")

print("\n4. 🔧 ОПТИМИЗАЦИЯ ЗАТРАТ:")
print("   • Снизить стоимость обслуживания через долгосрочные контракты")
print("   • Оптимизировать страховые расходы через групповые полисы")

print("\n5. 📊 МОНИТОРИНГ И АНАЛИТИКА:")
print("   • Внедрить систему мониторинга эффективности каждого велосипеда")
print("   • Регулярно обновлять категории на основе актуальных данных")

# ========== 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ==========
print("\n" + "=" * 100)
print("8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 100)

# Сохраняем все данные
bike_econ_df.to_csv('unit_economics_enhanced/bike_economics_detailed.csv', index=False)
category_summary.to_csv('unit_economics_enhanced/category_summary.csv')
sensitivity_df.to_csv('unit_economics_enhanced/sensitivity_analysis.csv', index=False)

# Создаем отчет
with open('unit_economics_enhanced/comprehensive_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n\n")

    f.write("ОБЩАЯ СТАТИСТИКА:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Всего велосипедов: {len(bike_econ_df)}\n")
    f.write(f"Средняя цена велосипеда: ${BIKE_PRICE_AVERAGE:.2f}\n")
    f.write(f"Общая прибыль: ${total_profit:,.2f}\n")
    f.write(f"Общий ROI: {overall_roi:.1f}%\n\n")

    f.write("РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ:\n")
    f.write("-" * 40 + "\n")
    for category in categories_order:
        if category in bike_econ_df['category'].unique():
            cat_data = bike_econ_df[bike_econ_df['category'] == category]
            f.write(f"{category}:\n")
            f.write(f"  Количество: {len(cat_data)}\n")
            f.write(f"  Средняя прибыль: ${cat_data['profit'].mean():.2f}\n")
            f.write(f"  Средний ROI: {cat_data['roi_percent'].mean():.1f}%\n\n")