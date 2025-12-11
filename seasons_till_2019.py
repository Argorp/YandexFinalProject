import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Настройки для красивого отображения
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)

print("=" * 70)
print("УЛУЧШЕНИЕ ЧИТАЕМОСТИ ДАННЫХ И АНАЛИЗ СЕЗОННОСТИ")
print("=" * 70)

# Загружаем очищенный датасет
df = pd.read_csv('cleaned_bike_sharing_data.csv')
print(f"Загружено записей: {len(df):,}")
print(f"Столбцов: {len(df.columns)}")

# ========== 1. УЛУЧШЕНИЕ ЧИТАЕМОСТИ ДАННЫХ ==========
print("\n" + "=" * 70)
print("1. УЛУЧШЕНИЕ ЧИТАЕМОСТИ ДАННЫХ")
print("=" * 70)

# 1.1. Преобразуем временные столбцы
df['starttime'] = pd.to_datetime(df['starttime'])
df['stoptime'] = pd.to_datetime(df['stoptime'])

# 1.2. Создаем более читаемые форматы
print("\n1.2. Создание читаемых форматов данных...")

# Время в понятном формате
df['start_datetime'] = df['starttime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df['stop_datetime'] = df['stoptime'].dt.strftime('%Y-%m-%d %H:%M:%S')


# Длительность в читаемом формате
def format_duration(seconds):
    """Преобразует секунды в читаемый формат"""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} сек"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes} мин {secs} сек"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours} ч {minutes} мин"


df['duration_readable'] = df['tripduration'].apply(format_duration)

# Возраст в годах
df['age_years'] = (datetime.now().year - df['birthyear']).astype(int)

# День недели русскими названиями
days_ru = {
    0: 'Понедельник', 1: 'Вторник', 2: 'Среда',
    3: 'Четверг', 4: 'Пятница', 5: 'Суббота', 6: 'Воскресенье'
}
df['day_of_week_ru'] = df['starttime'].dt.dayofweek.map(days_ru)

# Месяц русскими названиями
months_ru = {
    1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
    5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
    9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
}
df['month_ru'] = df['starttime'].dt.month.map(months_ru)


# Время суток с описанием
def get_time_period(hour):
    if 5 <= hour < 12:
        return 'Утро (5:00-11:59)'
    elif 12 <= hour < 17:
        return 'День (12:00-16:59)'
    elif 17 <= hour < 22:
        return 'Вечер (17:00-21:59)'
    else:
        return 'Ночь (22:00-4:59)'


df['time_period'] = df['starttime'].dt.hour.apply(get_time_period)


# Сезонность с русскими названиями
def get_season_ru(month):
    if month in [12, 1, 2]:
        return 'Зима'
    elif month in [3, 4, 5]:
        return 'Весна'
    elif month in [6, 7, 8]:
        return 'Лето'
    else:
        return 'Осень'


df['season_ru'] = df['starttime'].dt.month.apply(get_season_ru)


# Возрастные группы с описанием
def get_age_group_ru(age):
    if age < 18:
        return 'До 18 лет'
    elif 18 <= age < 25:
        return '18-24 года'
    elif 25 <= age < 35:
        return '25-34 года'
    elif 35 <= age < 45:
        return '35-44 года'
    elif 45 <= age < 55:
        return '45-54 года'
    elif 55 <= age < 65:
        return '55-64 года'
    else:
        return '65+ лет'


df['age_group_ru'] = df['age_years'].apply(get_age_group_ru)

# Тип пользователя на русском
df['usertype_ru'] = df['usertype'].map({
    'Subscriber': 'Подписчик',
    'Customer': 'Клиент'
}).fillna('Неизвестно')

# Пол на русском
df['gender_ru'] = df['gender'].map({
    'Male': 'Мужской',
    'Female': 'Женский'
}).fillna('Не указан')

# 1.3. Создаем сводные столбцы для быстрого анализа
df['year_month'] = df['starttime'].dt.strftime('%Y-%m')
df['week_year'] = df['starttime'].dt.strftime('%Y-W%W')
df['date'] = df['starttime'].dt.date

print("✓ Созданы читаемые форматы данных")

# ========== 2. АНАЛИЗ СЕЗОННОСТИ ==========
print("\n" + "=" * 70)
print("2. ПОДРОБНЫЙ АНАЛИЗ СЕЗОННОСТИ ПОЕЗДОК")
print("=" * 70)

# 2.1. Сезонность по месяцам
print("\n2.1. Сезонность по месяцам:")

# Создаем копию для агрегации по месяцам
monthly_stats = df.copy()
monthly_stats['month_num'] = monthly_stats['starttime'].dt.month

monthly_aggregate = monthly_stats.groupby(['month_num', 'month_ru', 'season_ru']).agg({
    'trip_id': 'count',
    'tripduration': 'mean',
    'usertype': lambda x: (x == 'Subscriber').mean() * 100
}).round(2)

monthly_aggregate.columns = ['total_trips', 'avg_duration', 'subscriber_pct']
monthly_aggregate = monthly_aggregate.reset_index()
monthly_aggregate = monthly_aggregate.sort_values('month_num')

print("\nСредняя активность по месяцам:")
for idx, row in monthly_aggregate.iterrows():
    print(f"  {row['month_ru']:10} ({row['season_ru']:6}): {row['total_trips']:6.0f} поездок, "
          f"длительность: {row['avg_duration'] / 60:5.1f} мин, "
          f"подписчики: {row['subscriber_pct']:5.1f}%")

# 2.2. Сезонность по временам года
print("\n2.2. Сезонность по временам года:")

seasonal_summary = df.groupby('season_ru').agg({
    'trip_id': 'count',
    'tripduration': 'mean',
    'usertype': lambda x: (x == 'Subscriber').mean() * 100,
    'starttime': lambda x: (x.max() - x.min()).days  # количество дней в сезоне в данных
}).round(2)

seasonal_summary.columns = ['total_trips', 'avg_duration', 'subscriber_pct', 'days_in_data']
seasonal_summary['avg_daily_trips'] = (seasonal_summary['total_trips'] / seasonal_summary['days_in_data']).round(0)
seasonal_summary = seasonal_summary.sort_values('total_trips', ascending=False)

print("\nАктивность по временам года:")
for season, row in seasonal_summary.iterrows():
    print(f"  {season:6}: {row['total_trips']:8,.0f} поездок, "
          f"среднедневно: {row['avg_daily_trips']:5.0f}, "
          f"длительность: {row['avg_duration'] / 60:5.1f} мин")

# 2.3. Сезонность по дням недели
print("\n2.3. Сезонность по дням недели:")

# Добавляем признак будний/выходной
df['is_weekend'] = df['starttime'].dt.dayofweek.isin([5, 6])

# Создаем агрегацию по дням недели
weekday_summary = df.groupby(['day_of_week_ru', 'is_weekend']).agg({
    'trip_id': 'count',
    'tripduration': 'mean',
    'usertype': lambda x: (x == 'Subscriber').mean() * 100
}).round(2)

weekday_summary.columns = ['total_trips', 'avg_duration', 'subscriber_pct']
weekday_summary = weekday_summary.reset_index()
weekday_summary = weekday_summary.sort_values('total_trips', ascending=False)

print("\nАктивность по дням недели:")
days_order = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']

for idx, row in weekday_summary.iterrows():
    is_weekend = "выходной" if row['is_weekend'] else "будний"
    print(f"  {row['day_of_week_ru']:12} ({is_weekend:8}): {row['total_trips']:8,.0f} поездок, "
          f"длительность: {row['avg_duration'] / 60:5.1f} мин")

# 2.4. Сезонность по времени суток
print("\n2.4. Сезонность по времени суток:")

hourly_summary = df.groupby('time_period').agg({
    'trip_id': 'count',
    'tripduration': 'mean',
    'usertype': lambda x: (x == 'Subscriber').mean() * 100
}).round(2)

hourly_summary.columns = ['total_trips', 'avg_duration', 'subscriber_pct']

# Сортируем по логическому порядку
time_order = ['Утро (5:00-11:59)', 'День (12:00-16:59)', 'Вечер (17:00-21:59)', 'Ночь (22:00-4:59)']
hourly_summary = hourly_summary.reindex(time_order)

print("\nАктивность по времени суток:")
for time_period, row in hourly_summary.iterrows():
    print(f"  {time_period:20}: {row['total_trips']:8,.0f} поездок, "
          f"длительность: {row['avg_duration'] / 60:5.1f} мин, "
          f"подписчики: {row['subscriber_pct']:5.1f}%")

# ========== 3. ВИЗУАЛИЗАЦИЯ СЕЗОННОСТИ ==========
print("\n" + "=" * 70)
print("3. ВИЗУАЛИЗАЦИЯ СЕЗОННОСТИ")
print("=" * 70)

# Создаем директорию для графиков
os.makedirs('seasonality_analysis', exist_ok=True)

# 3.1. График сезонности по месяцам
plt.figure(figsize=(14, 8))

# Подграфик 1: Количество поездок по месяцам
plt.subplot(2, 2, 1)
bars = plt.bar(monthly_aggregate['month_ru'], monthly_aggregate['total_trips'],
               color=plt.cm.viridis(np.linspace(0, 1, len(monthly_aggregate))))
plt.title('Количество поездок по месяцам', fontsize=14, fontweight='bold')
plt.xlabel('Месяц')
plt.ylabel('Количество поездок')
plt.xticks(rotation=45)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{int(height):,}', ha='center', va='bottom')

# Подграфик 2: Длительность поездок по месяцам
plt.subplot(2, 2, 2)
colors = ['#FF6B6B' if s == 'Зима' else '#4ECDC4' if s == 'Весна' else
'#45B7D1' if s == 'Лето' else '#96CEB4' for s in monthly_aggregate['season_ru']]
bars = plt.bar(monthly_aggregate['month_ru'], monthly_aggregate['avg_duration'] / 60, color=colors)
plt.title('Средняя длительность поездок по месяцам', fontsize=14, fontweight='bold')
plt.xlabel('Месяц')
plt.ylabel('Длительность (минуты)')
plt.xticks(rotation=45)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.1f}', ha='center', va='bottom')

# Подграфик 3: Активность по дням недели
plt.subplot(2, 2, 3)
# Переупорядочиваем данные по порядку дней
weekday_plot = weekday_summary.set_index('day_of_week_ru').reindex(days_order)
colors_weekday = ['#95a5a6' if i < 5 else '#e74c3c' for i in range(7)]
bars = plt.bar(weekday_plot.index, weekday_plot['total_trips'], color=colors_weekday)
plt.title('Активность по дням недели', fontsize=14, fontweight='bold')
plt.xlabel('День недели')
plt.ylabel('Количество поездок')
plt.xticks(rotation=45)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{int(height):,}', ha='center', va='bottom')

# Подграфик 4: Активность по времени суток
plt.subplot(2, 2, 4)
bars = plt.bar(hourly_summary.index, hourly_summary['total_trips'],
               color=plt.cm.coolwarm(np.linspace(0, 1, len(hourly_summary))))
plt.title('Активность по времени суток', fontsize=14, fontweight='bold')
plt.xlabel('Время суток')
plt.ylabel('Количество поездок')
plt.xticks(rotation=45)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{int(height):,}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('seasonality_analysis/seasonality_overview.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.2. Тепловая карта: день недели × час
plt.figure(figsize=(14, 8))
df['hour'] = df['starttime'].dt.hour
df['weekday_num'] = df['starttime'].dt.dayofweek

heatmap_data = df.groupby(['weekday_num', 'hour']).size().unstack(fill_value=0)
heatmap_data.index = [days_ru[i] for i in range(7)]

sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.0f',
            linewidths=.5, cbar_kws={'label': 'Количество поездок'})
plt.title('Тепловая карта: День недели × Час', fontsize=16, fontweight='bold')
plt.xlabel('Час дня')
plt.ylabel('День недели')
plt.tight_layout()
plt.savefig('seasonality_analysis/weekday_hour_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.3. График сезонности по годам (если данные за несколько лет)
if df['starttime'].dt.year.nunique() > 1:
    plt.figure(figsize=(12, 6))

    yearly_season = df.groupby(['starttime', 'season_ru']).size().reset_index()
    yearly_season['year'] = yearly_season['starttime'].dt.year

    seasonal_by_year = yearly_season.groupby(['year', 'season_ru'])[0].sum().unstack()

    seasonal_by_year.plot(kind='bar', figsize=(12, 6))
    plt.title('Сезонная активность по годам', fontsize=14, fontweight='bold')
    plt.xlabel('Год')
    plt.ylabel('Количество поездок')
    plt.legend(title='Сезон')
    plt.tight_layout()
    plt.savefig('seasonality_analysis/seasonality_by_year.png', dpi=300, bbox_inches='tight')
    plt.show()

# ========== 4. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ==========
print("\n" + "=" * 70)
print("4. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 70)

# 4.1. Сохраняем улучшенный датасет с читаемыми форматами
readable_columns = [
    'trip_id', 'start_datetime', 'stop_datetime', 'duration_readable',
    'bikeid', 'from_station_name', 'to_station_name', 'from_station_id',
    'to_station_id', 'usertype_ru', 'gender_ru', 'age_years', 'age_group_ru',
    'day_of_week_ru', 'month_ru', 'season_ru', 'time_period', 'year_month'
]

readable_df = df[readable_columns].copy()
readable_df.to_csv('bike_sharing_readable.csv', index=False, encoding='utf-8-sig')
print(f"✓ Читаемый датасет сохранен: bike_sharing_readable.csv ({len(readable_df):,} записей)")

# 4.2. Сохраняем аналитические таблицы
monthly_aggregate.to_csv('seasonality_analysis/monthly_analysis.csv', index=False, encoding='utf-8-sig')
seasonal_summary.to_csv('seasonality_analysis/seasonal_analysis.csv', encoding='utf-8-sig')
weekday_summary.to_csv('seasonality_analysis/weekday_analysis.csv', index=False, encoding='utf-8-sig')
hourly_summary.to_csv('seasonality_analysis/hourly_analysis.csv', encoding='utf-8-sig')

print("✓ Аналитические таблицы сохранены:")
print("  - monthly_analysis.csv (анализ по месяцам)")
print("  - seasonal_analysis.csv (анализ по сезонам)")
print("  - weekday_analysis.csv (анализ по дням недели)")
print("  - hourly_analysis.csv (анализ по времени суток)")

# 4.3. Создаем сводный отчет по сезонности
with open('seasonality_analysis/seasonality_report.txt', 'w', encoding='utf-8') as f:
    f.write("ОТЧЕТ ПО АНАЛИЗУ СЕЗОННОСТИ ПОЕЗДОК\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Всего поездок в анализе: {len(df):,}\n")
    f.write(f"Период данных: {df['starttime'].min().date()} - {df['starttime'].max().date()}\n\n")

    f.write("1. СЕЗОННОСТЬ ПО МЕСЯЦАМ:\n")
    f.write("-" * 40 + "\n")
    for idx, row in monthly_aggregate.iterrows():
        f.write(f"{row['month_ru']:10} | Поездок: {row['total_trips']:6.0f} | "
                f"Длит.: {row['avg_duration'] / 60:5.1f} мин | "
                f"Подписчики: {row['subscriber_pct']:5.1f}%\n")

    f.write("\n2. АКТИВНОСТЬ ПО СЕЗОНАМ:\n")
    f.write("-" * 40 + "\n")
    for season, row in seasonal_summary.iterrows():
        f.write(f"{season:6} | Всего: {row['total_trips']:8,.0f} | "
                f"В день: {row['avg_daily_trips']:5.0f} | "
                f"Длит.: {row['avg_duration'] / 60:5.1f} мин\n")

    f.write("\n3. АКТИВНОСТЬ ПО ДНЯМ НЕДЕЛИ:\n")
    f.write("-" * 40 + "\n")
    for idx, row in weekday_summary.iterrows():
        is_weekend = "выходной" if row['is_weekend'] else "будний"
        f.write(f"{row['day_of_week_ru']:12} ({is_weekend:8}) | "
                f"Поездок: {row['total_trips']:8,.0f} | "
                f"Длит.: {row['avg_duration'] / 60:5.1f} мин\n")

    f.write("\n4. АКТИВНОСТЬ ПО ВРЕМЕНИ СУТОК:\n")
    f.write("-" * 40 + "\n")
    for time_period, row in hourly_summary.iterrows():
        f.write(f"{time_period:20} | Поездок: {row['total_trips']:8,.0f} | "
                f"Длит.: {row['avg_duration'] / 60:5.1f} мин | "
                f"Подписчики: {row['subscriber_pct']:5.1f}%\n")

    f.write("\n5. ВЫВОДЫ И РЕКОМЕНДАЦИИ:\n")
    f.write("-" * 40 + "\n")

    # Автоматические выводы
    max_season = seasonal_summary['total_trips'].idxmax()
    min_season = seasonal_summary['total_trips'].idxmin()
    max_month = monthly_aggregate.loc[monthly_aggregate['total_trips'].idxmax(), 'month_ru']
    min_month = monthly_aggregate.loc[monthly_aggregate['total_trips'].idxmin(), 'month_ru']
    max_day = weekday_summary.loc[weekday_summary['total_trips'].idxmax(), 'day_of_week_ru']
    min_day = weekday_summary.loc[weekday_summary['total_trips'].idxmin(), 'day_of_week_ru']
    max_time = hourly_summary['total_trips'].idxmax()

    f.write(f"• Самый активный сезон: {max_season} ({seasonal_summary.loc[max_season, 'total_trips']:,.0f} поездок)\n")
    f.write(
        f"• Самый неактивный сезон: {min_season} ({seasonal_summary.loc[min_season, 'total_trips']:,.0f} поездок)\n")
    f.write(
        f"• Самый активный месяц: {max_month} ({monthly_aggregate.loc[monthly_aggregate['month_ru'] == max_month, 'total_trips'].values[0]:,.0f} поездок)\n")
    f.write(
        f"• Самый неактивный месяц: {min_month} ({monthly_aggregate.loc[monthly_aggregate['month_ru'] == min_month, 'total_trips'].values[0]:,.0f} поездок)\n")
    f.write(
        f"• Самый загруженный день: {max_day} ({weekday_summary.loc[weekday_summary['day_of_week_ru'] == max_day, 'total_trips'].values[0]:,.0f} поездок)\n")
    f.write(
        f"• Самый тихий день: {min_day} ({weekday_summary.loc[weekday_summary['day_of_week_ru'] == min_day, 'total_trips'].values[0]:,.0f} поездок)\n")
    f.write(f"• Пиковое время суток: {max_time} ({hourly_summary.loc[max_time, 'total_trips']:,.0f} поездок)\n")

    # Рекомендации
    f.write("\nРекомендации для оптимизации:\n")
    f.write("1. Увеличить количество велосипедов на станциях в пиковые часы\n")
    f.write("2. Проводить акции в низкий сезон для стимулирования спроса\n")
    f.write("3. Скорректировать график технического обслуживания на неактивные дни\n")
    f.write("4. Рассмотреть тарифную политику в зависимости от сезона\n")
    f.write("5. Оптимизировать распределение велосипедов между станциями\n")

print("✓ Отчет по сезонности сохранен: seasonality_analysis/seasonality_report.txt")

# 4.4. Создаем дашборд в Excel
try:
    with pd.ExcelWriter('seasonality_analysis/seasonality_dashboard.xlsx', engine='openpyxl') as writer:
        # Основные сводки
        summary_stats = pd.DataFrame({
            'Метрика': [
                'Всего поездок',
                'Период анализа',
                'Самый активный месяц',
                'Самый активный день недели',
                'Пиковое время суток',
                'Средняя длительность поездки',
                'Процент подписчиков',
                'Соотношение мужчины/женщины'
            ],
            'Значение': [
                f"{len(df):,}",
                f"{df['starttime'].min().date()} - {df['starttime'].max().date()}",
                max_month,
                max_day,
                max_time,
                f"{df['tripduration'].mean() / 60:.1f} минут",
                f"{(df['usertype'] == 'Subscriber').mean() * 100:.1f}%",
                f"{df['gender'].value_counts().get('Male', 0) / len(df) * 100:.1f}% / {df['gender'].value_counts().get('Female', 0) / len(df) * 100:.1f}%"
            ]
        })

        summary_stats.to_excel(writer, sheet_name='Сводка', index=False)
        monthly_aggregate.to_excel(writer, sheet_name='По месяцам', index=False)
        seasonal_summary.to_excel(writer, sheet_name='По сезонам')
        weekday_summary.to_excel(writer, sheet_name='По дням недели', index=False)
        hourly_summary.to_excel(writer, sheet_name='По времени суток')
        readable_df.head(1000).to_excel(writer, sheet_name='Пример данных', index=False)

    print("✓ Excel-дашборд сохранен: seasonality_analysis/seasonality_dashboard.xlsx")
except Exception as e:
    print(f"⚠️  Не удалось создать Excel-дашборд: {e}")

print("\n" + "=" * 70)
print("АНАЛИЗ СЕЗОННОСТИ ЗАВЕРШЕН!")
print("=" * 70)
print(f"\nСозданные файлы:")
print("1. bike_sharing_readable.csv - читаемый датасет")
print("2. seasonality_analysis/ - папка с анализом")
print("   ├── monthly_analysis.csv")
print("   ├── seasonal_analysis.csv")
print("   ├── weekday_analysis.csv")
print("   ├── hourly_analysis.csv")
print("   ├── seasonality_report.txt")
print("   ├── seasonality_overview.png")
print("   ├── weekday_hour_heatmap.png")

# Добавляем информацию о дополнительных графиках если они были созданы
if df['starttime'].dt.year.nunique() > 1:
    print("   └── seasonality_by_year.png")

print(f"\nКлючевые инсайты по сезонности:")
print(f"• Самый активный сезон: {max_season}")
print(f"• Пиковое время: {max_time}")
if min_season in seasonal_summary.index and max_season in seasonal_summary.index:
    diff_ratio = seasonal_summary.loc[max_season, 'total_trips'] / seasonal_summary.loc[min_season, 'total_trips']
    print(f"• Разница активности сезонов: {diff_ratio:.1f}x")

print(f"• Средняя длительность поездки: {df['tripduration'].mean() / 60:.1f} минут")
print(f"• Процент подписчиков: {(df['usertype'] == 'Subscriber').mean() * 100:.1f}%")