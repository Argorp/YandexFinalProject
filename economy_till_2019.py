import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Настройки
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print("=" * 80)
print("АНАЛИЗ ЮНИТ-ЭКОНОМИКИ: ОДИН ВЕЛОСИПЕД КАК БИЗНЕС-ЕДИНИЦА")
print("=" * 80)

# Загружаем очищенный датасет
df = pd.read_csv('cleaned_bike_sharing_data.csv')
df['starttime'] = pd.to_datetime(df['starttime'])

print(f"Всего поездок: {len(df):,}")
print(f"Уникальных велосипедов: {df['bikeid'].nunique():,}")
print(f"Период данных: {df['starttime'].min().date()} - {df['starttime'].max().date()}")
days_in_data = (df['starttime'].max() - df['starttime'].min()).days + 1
print(f"Дней в данных: {days_in_data}")

# ========== 1. ОПРЕДЕЛЕНИЕ ЮНИТА ==========
print("\n" + "=" * 80)
print("1. ОПРЕДЕЛЕНИЕ ЮНИТА И КЛЮЧЕВЫХ СВОЙСТВ")
print("=" * 80)

# Единица анализа: 1 велосипед
bike_analysis = df.groupby('bikeid').agg({
    'trip_id': 'count',
    'tripduration': ['sum', 'mean', 'median'],
    'starttime': ['min', 'max'],
    'from_station_id': 'nunique',
    'to_station_id': 'nunique',
    'usertype': lambda x: (x == 'Subscriber').mean()
}).round(2)

bike_analysis.columns = [
    'total_trips', 'total_duration_sec', 'avg_trip_duration', 'median_trip_duration',
    'first_use', 'last_use', 'unique_start_stations', 'unique_end_stations',
    'subscriber_ratio'
]

# Добавляем дополнительные метрики
bike_analysis['active_days'] = bike_analysis.apply(
    lambda row: (row['last_use'] - row['first_use']).days + 1, axis=1
)
bike_analysis['trips_per_day'] = bike_analysis['total_trips'] / bike_analysis['active_days']
bike_analysis['utilization_rate'] = bike_analysis['total_duration_sec'] / (bike_analysis['active_days'] * 24 * 3600)

print("\nКлючевые свойства юнита (1 велосипед):")
print("-" * 60)
print(f"• Среднее количество поездок на велосипед: {bike_analysis['total_trips'].mean():.0f}")
print(f"• Средняя длительность использования (дней): {bike_analysis['active_days'].mean():.0f}")
print(f"• Средняя утилизация (время в движении/общее время): {bike_analysis['utilization_rate'].mean() * 100:.2f}%")
print(f"• Среднее поездок в день: {bike_analysis['trips_per_day'].mean():.2f}")
print(f"• Процент подписчиков среди пользователей: {(bike_analysis['subscriber_ratio'].mean() * 100):.1f}%")

# ========== 2. ПРЕДПОЛОЖЕНИЯ ДЛЯ РАСЧЕТОВ ==========
print("\n" + "=" * 80)
print("2. ПРЕДПОЛОЖЕНИЯ ДЛЯ РАСЧЕТА ЭКОНОМИКИ")
print("=" * 80)

# Базовые предположения (могут быть скорректированы)
assumptions = {
    # Расходы на один велосипед
    'bike_cost': 800,  # USD - стоимость покупки велосипеда
    'bike_lifespan': 5,  # лет - срок службы
    'maintenance_per_trip': 0.15,  # USD - обслуживание за поездку
    'insurance_per_month': 5,  # USD - страховка в месяц
    'software_per_month': 2,  # USD - софт и приложение
    'storage_per_month': 3,  # USD - хранение и станция

    # Доходы
    'subscriber_fee_per_trip': 0.25,  # USD - плата подписчика за поездку
    'customer_fee_per_trip': 2.5,  # USD - плата клиента за поездку
    'subscriber_monthly_fee': 15,  # USD - месячная подписка
    'advertising_per_bike_per_month': 2,  # USD - реклама на велосипеде

    # Операционные показатели
    'avg_trips_per_day_low': 2,  # низкий сезон
    'avg_trips_per_day_high': 6,  # высокий сезон
    'subscriber_ratio': 0.7,  # доля подписчиков
    'operational_days_per_year': 300,  # дней в году с сервисом
}

print("\nБазовые предположения для расчета:")
print("-" * 60)
for key, value in assumptions.items():
    if isinstance(value, float):
        print(f"• {key}: {value:.2f}")
    else:
        print(f"• {key}: {value}")

# ========== 3. ФОРМУЛЫ ПРИБЫЛИ И РАСХОДОВ ==========
print("\n" + "=" * 80)
print("3. ФОРМУЛЫ ПРИБЫЛИ И РАСХОДОВ")
print("=" * 80)


def calculate_bike_economics(assumptions, season_factor=1.0):
    """Расчет экономики одного велосипеда"""

    # Средние поездки в день с учетом сезонности
    avg_trips_per_day = (assumptions['avg_trips_per_day_low'] +
                         (assumptions['avg_trips_per_day_high'] - assumptions['avg_trips_per_day_low']) * season_factor)

    # ГОДОВЫЕ РАСХОДЫ
    # Амортизация велосипеда
    depreciation_per_year = assumptions['bike_cost'] / assumptions['bike_lifespan']

    # Обслуживание
    maintenance_per_year = (avg_trips_per_day * assumptions['operational_days_per_year'] *
                            assumptions['maintenance_per_trip'])

    # Фиксированные расходы (в месяц)
    fixed_costs_per_year = (assumptions['insurance_per_month'] +
                            assumptions['software_per_month'] +
                            assumptions['storage_per_month']) * 12

    total_costs_per_year = depreciation_per_year + maintenance_per_year + fixed_costs_per_year

    # ГОДОВЫЕ ДОХОДЫ
    total_trips_per_year = avg_trips_per_day * assumptions['operational_days_per_year']

    # Доходы от поездок
    subscriber_trips = total_trips_per_year * assumptions['subscriber_ratio']
    customer_trips = total_trips_per_year * (1 - assumptions['subscriber_ratio'])

    trip_revenue = (subscriber_trips * assumptions['subscriber_fee_per_trip'] +
                    customer_trips * assumptions['customer_fee_per_trip'])

    # Доходы от подписок
    subscriber_revenue = (assumptions['subscriber_monthly_fee'] * 12 *
                          (avg_trips_per_day * 30 * assumptions['subscriber_ratio'] / 10))  # упрощенная модель

    # Прочие доходы
    other_revenue = assumptions['advertising_per_bike_per_month'] * 12

    total_revenue_per_year = trip_revenue + subscriber_revenue + other_revenue

    # ПРИБЫЛЬ
    profit_per_year = total_revenue_per_year - total_costs_per_year
    profit_margin = (profit_per_year / total_revenue_per_year * 100) if total_revenue_per_year > 0 else 0

    # Расчет окупаемости
    payback_period_months = (assumptions['bike_cost'] / (profit_per_year / 12)) if profit_per_year > 0 else float('inf')

    return {
        'avg_trips_per_day': avg_trips_per_day,
        'total_trips_per_year': total_trips_per_year,
        'depreciation_per_year': depreciation_per_year,
        'maintenance_per_year': maintenance_per_year,
        'fixed_costs_per_year': fixed_costs_per_year,
        'total_costs_per_year': total_costs_per_year,
        'trip_revenue': trip_revenue,
        'subscriber_revenue': subscriber_revenue,
        'other_revenue': other_revenue,
        'total_revenue_per_year': total_revenue_per_year,
        'profit_per_year': profit_per_year,
        'profit_margin': profit_margin,
        'payback_period_months': payback_period_months
    }


# Расчет для средней сезонности
base_economics = calculate_bike_economics(assumptions, season_factor=0.5)
print("\nГодовая экономика одного велосипеда (средняя сезонность):")
print("-" * 60)
for key, value in base_economics.items():
    if isinstance(value, float):
        if 'margin' in key or 'period' in key:
            print(f"• {key}: {value:.1f}" + ("%" if 'margin' in key else " мес"))
        else:
            print(f"• {key}: ${value:,.2f}")

# ========== 4. АНАЛИЗ СЕЗОННОСТИ ==========
print("\n" + "=" * 80)
print("4. АНАЛИЗ СЕЗОННОСТИ ДЛЯ ЮНИТ-ЭКОНОМИКИ")
print("=" * 80)

# Анализ фактической сезонности по велосипедам
df['month'] = df['starttime'].dt.month
df['season'] = df['month'].apply(lambda x: 'Зима' if x in [12, 1, 2] else
'Весна' if x in [3, 4, 5] else
'Лето' if x in [6, 7, 8] else 'Осень')

# Группируем по велосипедам и сезонам
seasonal_bike_stats = df.groupby(['bikeid', 'season']).agg({
    'trip_id': 'count',
    'tripduration': 'sum',
    'usertype': lambda x: (x == 'Subscriber').mean()
}).reset_index()

seasonal_bike_stats = seasonal_bike_stats.rename(columns={
    'trip_id': 'trips',
    'tripduration': 'total_duration'
})

# Агрегируем по сезонам
seasonal_summary = seasonal_bike_stats.groupby('season').agg({
    'trips': ['mean', 'std', 'count'],
    'total_duration': 'mean',
    'usertype': 'mean'
}).round(2)

seasonal_summary.columns = ['avg_trips', 'std_trips', 'bike_count',
                            'avg_duration_sec', 'subscriber_ratio']

# Определяем сезонные коэффициенты
summer_trips = seasonal_summary.loc['Лето', 'avg_trips'] if 'Лето' in seasonal_summary.index else 0
winter_trips = seasonal_summary.loc['Зима', 'avg_trips'] if 'Зима' in seasonal_summary.index else 0

if summer_trips > 0 and winter_trips > 0:
    seasonal_ratio = summer_trips / winter_trips
    print(f"\nФактическая сезонность (Лето/Зима): {seasonal_ratio:.2f}x")
else:
    # Используем средние значения по сезонам
    seasonal_avg = seasonal_summary['avg_trips'].mean()
    seasonal_ratio = 2.0  # консервативная оценка
    print(f"\nСредняя активность по сезонам: {seasonal_avg:.1f} поездок/велосипед/сезон")

print("\nСезонная статистика по велосипедам:")
print("-" * 60)
seasons_order = ['Зима', 'Весна', 'Лето', 'Осень']
for season in seasons_order:
    if season in seasonal_summary.index:
        row = seasonal_summary.loc[season]
        print(f"{season:6}: {row['avg_trips']:5.1f} поездок, "
              f"длительность: {row['avg_duration_sec'] / 3600:5.1f} ч, "
              f"подписчики: {row['subscriber_ratio'] * 100:4.1f}%")

# ========== 5. ГРАФИЧЕСКИЙ АНАЛИЗ ==========
print("\n" + "=" * 80)
print("5. ГРАФИЧЕСКИЙ АНАЛИЗ ЮНИТ-ЭКОНОМИКИ")
print("=" * 80)

# Создаем директорию для графиков
import os

os.makedirs('unit_economics', exist_ok=True)

# 5.1. Распределение активности велосипедов
plt.figure(figsize=(14, 10))

# Подграфик 1: Распределение поездок на велосипед
plt.subplot(2, 2, 1)
top_bikes = bike_analysis.nlargest(30, 'total_trips')
plt.barh(top_bikes.index.astype(str)[-15:], top_bikes['total_trips'][-15:])
plt.xlabel('Количество поездок')
plt.title('Топ-15 самых активных велосипедов')
plt.gca().invert_yaxis()

# Подграфик 2: Распределение утилизации
plt.subplot(2, 2, 2)
utilization_bins = pd.cut(bike_analysis['utilization_rate'] * 100,
                          bins=[0, 1, 2, 5, 10, 20, 100])
utilization_dist = utilization_bins.value_counts().sort_index()
plt.bar([str(x) for x in utilization_dist.index], utilization_dist.values)
plt.xlabel('Утилизация (%)')
plt.ylabel('Количество велосипедов')
plt.title('Распределение утилизации велосипедов')
plt.xticks(rotation=45)

# Подграфик 3: Сезонность доходов
plt.subplot(2, 2, 3)
seasons = ['Зима', 'Весна', 'Лето', 'Осень']
season_factors = [0.0, 0.3, 1.0, 0.5]  # факторы сезонности
seasonal_profits = []

for factor in season_factors:
    economics = calculate_bike_economics(assumptions, factor)
    seasonal_profits.append(economics['profit_per_year'])

plt.bar(seasons, seasonal_profits, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
plt.xlabel('Сезон')
plt.ylabel('Годовая прибыль ($)')
plt.title('Сезонность прибыли на один велосипед')

# Добавляем значения на столбцы
for i, v in enumerate(seasonal_profits):
    plt.text(i, v + max(seasonal_profits) * 0.02, f'${v:,.0f}',
             ha='center', va='bottom')

# Подграфик 4: Структура затрат и доходов
plt.subplot(2, 2, 4)
costs = [
    base_economics['depreciation_per_year'],
    base_economics['maintenance_per_year'],
    base_economics['fixed_costs_per_year']
]
revenues = [
    base_economics['trip_revenue'],
    base_economics['subscriber_revenue'],
    base_economics['other_revenue']
]

cost_labels = ['Амортизация', 'Обслуживание', 'Фикс.расходы']
revenue_labels = ['Поездки', 'Подписки', 'Реклама']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.pie(costs, labels=cost_labels, autopct='%1.1f%%', startangle=90)
ax1.set_title('Структура затрат')
ax2.pie(revenues, labels=revenue_labels, autopct='%1.1f%%', startangle=90)
ax2.set_title('Структура доходов')

plt.tight_layout()
plt.savefig('unit_economics/bike_analysis_overview.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.2. Анализ окупаемости
plt.figure(figsize=(12, 8))

# Сценарии окупаемости
scenarios = [
    {'name': 'Пессимистичный', 'trips_factor': 0.7, 'subscriber_ratio': 0.5},
    {'name': 'Базовый', 'trips_factor': 1.0, 'subscriber_ratio': 0.7},
    {'name': 'Оптимистичный', 'trips_factor': 1.3, 'subscriber_ratio': 0.8}
]

months = list(range(1, 37))  # 3 года
cumulative_profits = []

for scenario in scenarios:
    modified_assumptions = assumptions.copy()
    modified_assumptions['avg_trips_per_day_low'] *= scenario['trips_factor']
    modified_assumptions['avg_trips_per_day_high'] *= scenario['trips_factor']
    modified_assumptions['subscriber_ratio'] = scenario['subscriber_ratio']

    economics = calculate_bike_economics(modified_assumptions, 0.5)
    monthly_profit = economics['profit_per_year'] / 12

    cumulative = []
    total = 0
    for month in months:
        total += monthly_profit
        cumulative.append(total)

    cumulative_profits.append(cumulative)
    payback_month = next((i for i, x in enumerate(cumulative) if x >= assumptions['bike_cost']), None)

    plt.plot(months, cumulative,
             label=f"{scenario['name']} (окупаемость: {payback_month + 1 if payback_month else '>36'} мес)",
             linewidth=2)

plt.axhline(y=assumptions['bike_cost'], color='r', linestyle='--', alpha=0.5, label='Стоимость велосипеда')
plt.axvline(x=12, color='g', linestyle='--', alpha=0.5, label='1 год')
plt.axvline(x=24, color='g', linestyle='--', alpha=0.3, label='2 года')

plt.xlabel('Месяцы')
plt.ylabel('Накопленная прибыль ($)')
plt.title('Анализ окупаемости велосипеда по сценариям')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('unit_economics/payback_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.3. Чувствительность к ключевым параметрам
plt.figure(figsize=(14, 6))

# Анализ чувствительности прибыли к ключевым параметрам
parameters = {
    'Среднее поездок в день': np.linspace(1, 8, 15),
    'Стоимость обслуживания ($/поездка)': np.linspace(0.05, 0.3, 15),
    'Доля подписчиков (%)': np.linspace(0.3, 0.9, 15),
    'Стоимость велосипеда ($)': np.linspace(500, 1200, 15)
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (param_name, param_range) in enumerate(parameters.items()):
    ax = axes[idx // 2, idx % 2]
    profits = []

    for value in param_range:
        modified_assumptions = assumptions.copy()

        if 'поездок' in param_name:
            modified_assumptions['avg_trips_per_day_low'] = value * 0.7
            modified_assumptions['avg_trips_per_day_high'] = value * 1.3
        elif 'обслуживания' in param_name:
            modified_assumptions['maintenance_per_trip'] = value
        elif 'подписчиков' in param_name:
            modified_assumptions['subscriber_ratio'] = value
        elif 'велосипеда' in param_name:
            modified_assumptions['bike_cost'] = value

        economics = calculate_bike_economics(modified_assumptions, 0.5)
        profits.append(economics['profit_per_year'])

    ax.plot(param_range, profits, linewidth=2, color='#2c3e50')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Годовая прибыль ($)')
    ax.set_title(f'Чувствительность к {param_name.lower()}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('unit_economics/sensitivity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 6. ВЫВОДЫ И ЦИФРЫ ==========
print("\n" + "=" * 80)
print("6. ВЫВОДЫ И КЛЮЧЕВЫЕ ЦИФРЫ")
print("=" * 80)

# Расчет для различных сценариев
print("\nКлючевые метрики юнит-экономики:")
print("-" * 60)

scenario_results = []
for scenario in scenarios:
    modified_assumptions = assumptions.copy()
    modified_assumptions['avg_trips_per_day_low'] *= scenario['trips_factor']
    modified_assumptions['avg_trips_per_day_high'] *= scenario['trips_factor']
    modified_assumptions['subscriber_ratio'] = scenario['subscriber_ratio']

    economics = calculate_bike_economics(modified_assumptions, 0.5)
    scenario_results.append({
        'Сценарий': scenario['name'],
        'Годовая прибыль': economics['profit_per_year'],
        'Маржа': economics['profit_margin'],
        'Окупаемость (мес)': economics['payback_period_months'],
        'ROI за 3 года (%)': (economics['profit_per_year'] * 3 / assumptions['bike_cost'] * 100)
    })

results_df = pd.DataFrame(scenario_results)
print(results_df.to_string(index=False))

# Фактические данные из анализа
print("\nФактические данные из вашего датасета:")
print("-" * 60)
print(f"• Всего велосипедов: {df['bikeid'].nunique():,}")
print(f"• Всего поездок: {len(df):,}")
print(f"• Среднее поездок на велосипед: {bike_analysis['total_trips'].mean():.0f}")
print(f"• Максимум поездок на один велосипед: {bike_analysis['total_trips'].max():.0f}")
print(f"• Минимум поездок на один велосипед: {bike_analysis['total_trips'].min():.0f}")
print(f"• Средняя утилизация: {bike_analysis['utilization_rate'].mean() * 100:.2f}%")
print(f"• Медианное время жизни велосипеда: {bike_analysis['active_days'].median():.0f} дней")

# Расчет фактической экономики на основе реальных данных
actual_avg_trips_per_day = bike_analysis['trips_per_day'].median()
actual_subscriber_ratio = df['usertype'].value_counts(normalize=True).get('Subscriber', 0)

print(f"\nФактические показатели из данных:")
print("-" * 60)
print(f"• Среднее поездок в день (медиана): {actual_avg_trips_per_day:.2f}")
print(f"• Доля подписчиков: {actual_subscriber_ratio * 100:.1f}%")

# Пересчет с фактическими данными
if actual_avg_trips_per_day > 0:
    modified_assumptions = assumptions.copy()
    modified_assumptions['avg_trips_per_day_low'] = actual_avg_trips_per_day * 0.7
    modified_assumptions['avg_trips_per_day_high'] = actual_avg_trips_per_day * 1.3
    modified_assumptions['subscriber_ratio'] = actual_subscriber_ratio

    actual_economics = calculate_bike_economics(modified_assumptions, 0.5)

    print(f"\nПрогноз на основе фактических данных:")
    print("-" * 60)
    print(f"• Годовая прибыль на велосипед: ${actual_economics['profit_per_year']:,.2f}")
    print(f"• Маржа прибыли: {actual_economics['profit_margin']:.1f}%")
    print(f"• Период окупаемости: {actual_economics['payback_period_months']:.1f} месяцев")
    print(f"• ROI за 3 года: {(actual_economics['profit_per_year'] * 3 / assumptions['bike_cost'] * 100):.1f}%")

# Рекомендации
print("\n" + "=" * 80)
print("РЕКОМЕНДАЦИИ ДЛЯ БИЗНЕСА")
print("=" * 80)

print("\n1. Оптимизация использования велосипедов:")
print(f"   • Текущая утилизация: {bike_analysis['utilization_rate'].mean() * 100:.1f}%")
print(f"   • Целевая утилизация: 8-12% (увеличение в {(0.1 / bike_analysis['utilization_rate'].mean()):.1f} раз)")
print(f"   • Необходимо увеличить поездки в день с {actual_avg_trips_per_day:.2f} до 3-4")

print("\n2. Управление сезонностью:")
print("   • Пик сезона: Лето (активность в 2-3 раза выше)")
print("   • Низкий сезон: Зима (требуются промо-акции)")
print("   • Рекомендация: ввести динамическое ценообразование по сезонам")

print("\n3. Финансовые рекомендации:")
print(
    f"   • Минимальная цена поездки для безубыточности: ${(base_economics['total_costs_per_year'] / base_economics['total_trips_per_year']):.2f}")
print(f"   • Критическая точка безубыточности: {base_economics['total_trips_per_year']:.0f} поездок/год")
print(f"   • Рекомендуемый запас прочности: +20% к целевым показателям")

print("\n4. Инвестиционные выводы:")
print(f"   • Окупаемость инвестиций: {base_economics['payback_period_months']:.1f} месяцев")
print(f"   • ROI за 3 года: {(base_economics['profit_per_year'] * 3 / assumptions['bike_cost'] * 100):.1f}%")
print("   • Рекомендация: проект привлекателен при ROI > 100% за 3 года")

# ========== 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ==========
print("\n" + "=" * 80)
print("7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 80)

# Сохраняем аналитические таблицы
bike_analysis.to_csv('unit_economics/bike_performance_stats.csv')
seasonal_summary.to_csv('unit_economics/seasonal_bike_stats.csv')

# Сохраняем сводный отчет
with open('unit_economics/unit_economics_summary.txt', 'w', encoding='utf-8') as f:
    f.write("ОТЧЕТ ПО ЮНИТ-ЭКОНОМИКЕ: ОДИН ВЕЛОСИПЕД\n")
    f.write("=" * 70 + "\n\n")

    f.write("КЛЮЧЕВЫЕ ВЫВОДЫ:\n")
    f.write("-" * 40 + "\n")
    f.write(f"1. Годовая прибыль на велосипед: ${base_economics['profit_per_year']:,.2f}\n")
    f.write(f"2. Маржа прибыли: {base_economics['profit_margin']:.1f}%\n")
    f.write(f"3. Период окупаемости: {base_economics['payback_period_months']:.1f} месяцев\n")
    f.write(f"4. ROI за 3 года: {(base_economics['profit_per_year'] * 3 / assumptions['bike_cost'] * 100):.1f}%\n")
    f.write(f"5. Точка безубыточности: {base_economics['total_trips_per_year']:.0f} поездок/год\n\n")

    f.write("ФАКТИЧЕСКИЕ ПОКАЗАТЕЛИ ИЗ ДАННЫХ:\n")
    f.write("-" * 40 + "\n")
    f.write(f"• Среднее поездок на велосипед: {bike_analysis['total_trips'].mean():.0f}\n")
    f.write(f"• Средняя утилизация: {bike_analysis['utilization_rate'].mean() * 100:.1f}%\n")
    f.write(f"• Доля подписчиков: {actual_subscriber_ratio * 100:.1f}%\n\n")

    f.write("РЕКОМЕНДАЦИИ:\n")
    f.write("-" * 40 + "\n")
    f.write("1. Увеличить утилизацию до 8-12%\n")
    f.write("2. Внедрить динамическое ценообразование\n")
    f.write("3. Стимулировать подписочную модель\n")
    f.write("4. Оптимизировать распределение велосипедов\n")

print("✓ Сохранены файлы анализа:")
print("  - unit_economics/bike_performance_stats.csv")
print("  - unit_economics/seasonal_bike_stats.csv")
print("  - unit_economics/unit_economics_summary.txt")
print("  - unit_economics/bike_analysis_overview.png")
print("  - unit_economics/payback_analysis.png")
print("  - unit_economics/sensitivity_analysis.png")

print("\n" + "=" * 80)
print("АНАЛИЗ ЮНИТ-ЭКОНОМИКИ ЗАВЕРШЕН!")
print("=" * 80)