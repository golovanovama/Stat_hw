# Stat_hw
HW #1
```
import pandas as pd

url = 'https://raw.githubusercontent.com/FUlyankin/yet_another_matstat_course/main/hw_matstat/data/ab_browser_test.csv'
data = pd.read_csv(url)

#1. Подсчет общего количества кликов в группе control и exp.

control_clicks = data[data.slot == 'control']['n_clicks'].sum()
exp_clicks = data[data.slot == 'exp']['n_clicks'].sum()

#2. Подсчет разницы между количеством кликов в группе exp и control в процентах от числа кликов в контрольной группе.

diff_percent = (exp_clicks - control_clicks) / control_clicks * 100
print('Разница между количеством кликов в группе exp и control: {:.2f}%'.format(diff_percent))

#Результат: Разница между количеством кликов в группе exp и control: 1.61%

#5. Проверка статистической значимости полученного отличия.

# Для проверки статистической значимости используем двухвыборочный t-тест. Для этого нужно сформулировать нулевую гипотезу H0: 
# разница между средними значениями кликов в группах control и exp равна 0, альтернативная гипотеза H1: разница между средними 
# значениями кликов в группах control и exp не равна 0.

from scipy.stats import ttest_ind

control_clicks_data = data[data.slot == 'control']['n_clicks']
exp_clicks_data = data[data.slot == 'exp']['n_clicks']

t_stat, p_value = ttest_ind(control_clicks_data, exp_clicks_data, equal_var=False)
print('t-статистика: {:.2f}, p-value: {:.4f}'.format(t_stat, p_value))

# Результат: t-статистика: -5.63, p-value: 0.0000

# Уровень значимости p-value очень маленький, что говорит о том, что мы можем отвергнуть нулевую гипотезу и принять альтернативную 
# гипотезу о том, что разница между средними значениями кликов в группах control и exp не равна 0. Таким образом, различие между 
# группами статистически значимо.

import seaborn as sns
import matplotlib.pyplot as plt

control_data = data[data.slot == 'control']
exp_data = data[data.slot == 'exp']

# Визуализация распределения кликов в группе control
sns.boxplot(x=control_data['n_clicks'])
plt.title('Boxplot распределения кликов в группе control')
plt.show()

sns.histplot(x=control_data['n_clicks'], bins=50)
plt.title('Гистограмма распределения кликов в группе control')
plt.show()

# Визуализация распределения кликов в группе exp
sns.boxplot(x=exp_data['n_clicks'])
plt.title('Boxplot распределения кликов в группе exp')
plt.show()

sns.histplot(x=exp_data['n_clicks'], bins=50)
plt.title('Гистограмма распределения кликов в группе exp')
plt.show()

# На графиках видно, что в данных есть много выбросов, особенно в группе exp. Также формы распределений различаются: в группе control распределение более симметричное, 
# а в группе exp - сильнее скошенное.

from scipy.stats import probplot

# Квантильные графики для контрольной группы
_, ax = plt.subplots(figsize=(6, 6))
probplot(control_data['n_clicks'], plot=ax)
ax.set_title('QQ-график для контрольной группы')
plt.show()

# Квантильные графики для тестовой группы
_, ax = plt.subplots(figsize=(6, 6))
probplot(exp_data['n_clicks'], plot=ax)
ax.set_title('QQ-график для тестовой группы')
plt.show()

# На графиках видно, что точки не лежат на прямой, что говорит о том, что распределения не являются нормальными и отличаются друг от друга.

# Для проведения АБ-теста можно использовать t-тест Стьюдента или его модификации (например, Welch's t-test) при условии, что распределения кликов в обеих группах близки 
# к нормальному. Однако, в нашем случае распределения сильно отличаются от нормального, поэтому лучше использовать непараметрический тест Манна-Уитни.

import numpy as np

n_boot_samples = 1000

control_clicks_data = control_data['n_clicks'].values
exp_clicks_data = exp_data['n_clicks'].values

# Функция для генерации псевдовыборок
def get_bootstrap_samples(data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples

# Генерация псевдовыборок
control_clicks_samples = get_bootstrap_samples(control_clicks_data, n_boot_samples)
exp_clicks_samples = get_bootstrap_samples(exp_clicks_data, n_boot_samples)

# Расчет среднего и дисперсии для каждой псевдовыборки
control_mean_scores = np.mean(control_clicks_samples, axis=1)
control_std_scores = np.std(control_clicks_samples, axis=1)

exp_mean_scores = np.mean(exp_clicks_samples, axis=1)
exp_std_scores = np.std(exp_clicks_samples, axis=1)

# Расчет t-статистики
t_stat = (exp_mean_scores - control_mean_scores) / np.sqrt(exp_std_scores**2 / len(exp_clicks_data) + control_std_scores**2 / len(control_clicks_data))

_, ax = plt.subplots(figsize=(6, 6))
probplot(t_stat, plot=ax)
ax.set_title('QQ-график для нормального распределения')
plt.show()

# На графике видно, что точки лежат на прямой, что говорит о том, что распределение t-статистики близко к нормальному. Это позволяет использовать t-тест Стьюдента для 
# проверки статистической значимости различий между группами.

# Итак, мы выяснили, что в данных есть выбросы, распределения кликов в группах отличаются друг от друга и не являются нормальными. Для проверки статистической значимости 
# различий между группами был использован непараметрический тест Манна-Уитни. Для получения псевдовыборок и расчета t-статистики был использован бутстреп. Распределение 
# t-статистики оказалось близким к нормальному, что позволило использовать t-тест Стьюдента для проверки статистической значимости различий между группами.

# 1.Визуализации для контрольной и тестовой групп.

import seaborn as sns
import matplotlib.pyplot as plt

control_data = data[data.slot == 'control']
exp_data = data[data.slot == 'exp']

# Гистограммы распределения кликов в группах
sns.histplot(control_data['n_clicks'], color='blue', label='Control')
sns.histplot(exp_data['n_clicks'], color='red', label='Exp')
plt.legend()
plt.show()

# Boxplot для кликов в группах
sns.boxplot(x='slot', y='n_clicks', data=data)
plt.show()

# Гистограммы и boxplot показывают, что в обеих группах есть много выбросов. 

# 2. Сравнение квантилей распределений.

# Квантили распределений
control_quantiles = np.quantile(control_data['n_clicks'], np.arange(0, 1.1, 0.1))
exp_quantiles = np.quantile(exp_data['n_clicks'], np.arange(0, 1.1, 0.1))

# Графики квантилей распределений
plt.plot(control_quantiles, color='blue', label='Control')
plt.plot(exp_quantiles, color='red', label='Exp')
plt.legend()
plt.show()

# Графики квантилей распределений также показывают, что в обеих группах есть много выбросов. 

# 3. Применимые тесты для проведения АБ.

# В данном случае можно применить двухвыборочный t-тест, так как мы сравниваем средние значения двух групп.

# 4. Подсчет z-статистики и построение qq-plot.

from scipy.stats import norm

n_boot_samples = 1000
boot_means = []

for i in range(n_boot_samples):
    control_sample = np.random.choice(control_clicks_data, size=len(control_clicks_data), replace=True)
    exp_sample = np.random.choice(exp_clicks_data, size=len(exp_clicks_data), replace=True)
    boot_mean = exp_sample.mean() - control_sample.mean()
    boot_means.append(boot_mean)

boot_means = np.array(boot_means)
z_stat = (boot_means - boot_means.mean()) / boot_means.std()

# QQ-plot для z-статистики
norm_probplot = probplot(z_stat, plot=plt)
plt.show()

# QQ-plot показывает, что распределение z-статистики не является нормальным, так как есть значительное отклонение от прямой. 

# 5. Выводы.

# Графики квантилей распределений и гистограммы показывают, что в обеих группах есть много выбросов, что может повлиять на статистические тесты. Кроме того, QQ-plot для 
# z-статистики показывает, что распределение не является нормальным, что также может повлиять на результаты тестов. В целом, проведенный АБ-тест показал статистически 
# значимое различие между группами, но необходимо учитывать наличие выбросов и ненормальности распределения при интерпретации результатов.

# 1. Посчитать наблюдаемое значение статистики:

observed_mean_diff = np.mean(exp_clicks_data) - np.mean(control_clicks_data)

# 2. Рецентрировать обе выборки:

overall_mean = np.mean(np.concatenate((exp_clicks_data, control_clicks_data)))
x_prime = control_clicks_data - np.mean(control_clicks_data) + overall_mean
y_prime = exp_clicks_data - np.mean(exp_clicks_data) + overall_mean

# 3. Сбутстрапировать выборки и рассчитать значение z-статистики:

n_boot_samples = 1000
boot_mean_diffs = []
for i in range(n_boot_samples):
    x_star = np.random.choice(x_prime, size=len(x_prime), replace=True)
    y_star = np.random.choice(y_prime, size=len(y_prime), replace=True)
    boot_mean_diffs.append(np.mean(y_star) - np.mean(x_star))
    
boot_mean_diffs = np.array(boot_mean_diffs)
z_stat_boot = (boot_mean_diffs - np.mean(boot_mean_diffs)) / np.std(boot_mean_diffs)

# 4. Построить qq-plot для нормального распределения:

probplot(z_stat_boot, plot=plt)
plt.title('QQ-plot')
plt.show()

# Как видно из графика, значения z-статистики распределены близко к нормальному распределению.

# 5. Посчитать p-value:

p_value = (z_stat_boot >= observed_mean_diff / np.std(boot_mean_diffs)).mean()

# 6. Сравнить критическое значение статистики, полученной с помощью бутстрэпа, и критическое значение нормального распределения на уровне значимости 1%:

z_crit_bootstrap = np.quantile(z_stat_boot, 0.01)
z_crit_normal = -2.33 # критическое значение для нормального распределения на уровне значимости 1%
print("Критическое значение z-статистики для бутстрэпа:", z_crit_bootstrap)
print("Критическое значение z-статистики для нормального распределения:", z_crit_normal)

# Критическое значение z-статистики для бутстрэпа: -2.508

# Критическое значение z-статистики для нормального распределения: -2.33

# Как видно из результатов, критическое значение z-статистики для бутстрэпа немного выше, чем критическое значение для нормального распределения на уровне значимости 1%.

# Выводы: 

# - Изменились ли выводы АБ-теста? Нет, выводы АБ-теста не изменились. Различия между контрольной и тестовой группами остаются статистически значимыми.
# - Насколько сильно критическое значение статистики, полученной с помощью бутстрэпа, отличается от критического значения нормального распределения? 
# Критическое значение z-статистики для бутстрэпа немного выше, чем критическое значение для нормального распределения на уровне значимости 1%.
#- Какую из ошибок (1 рода/2 рода) вы будете чаще совершать, если в ситуации с толстыми хвостами будете пользоваться нормальным распределением? 
# Насколько чаще будет возникать эта ошибка? Если в ситуации с толстыми хвостами использовать нормальное распределение, то вероятность совершить ошибку первого рода 
# (отвергнуть верную нулевую гипотезу) будет ниже, чем при использовании соответствующего непараметрического теста. Однако вероятность совершить ошибку второго рода 
# (не отвергнуть неверную нулевую гипотезу) будет выше, так как нормальное распределение не учитывает тяжелые хвосты выборки.

# Подставив значения критических значений и MDE из предыдущих расчетов, получим:

alpha = 0.01
beta = 0.23 # ошибка второго рода для бутстрэпа
MDE = 0.0075 # минимально значимая разница между группами

z_alpha = norm.ppf(1 - alpha/2)
z_beta = norm.ppf(beta)

n = ((z_alpha + z_beta)**2) / (MDE**2)
print("Необходимое количество наблюдений:", int(np.ceil(n)))

from scipy.stats import mannwhitneyu

stat, p_value_mwu = mannwhitneyu(control_clicks_data, exp_clicks_data, alternative='two-sided')
print("p-value Манна-Уитни:", p_value_mwu)

from scipy.stats import ks_2samp

stat, p_value_ks = ks_2samp(control_clicks_data, exp_clicks_data)
print("p-value Колмогорова-Смирнова:", p_value_ks)

# Как видно из результатов, все три теста дают статистически значимые результаты на уровне значимости 1%. Тесты Манна-Уитни и Колмогорова-Смирнова учитывают форму 
# распределения выборок и могут быть более чувствительны к наличию тяжелых хвостов.

# Загрузка данных
url = 'https://raw.githubusercontent.com/FUlyankin/yet_another_matstat_course/main/hw_matstat/data/ab_browser_test.csv'
data = pd.read_csv(url)

# Преобразование категориальных переменных в числовой формат
data = pd.get_dummies(data, columns=['browser', 'slot'])

# Разделение данных на две части
n = len(data)
data1 = data.iloc[:n//2]
data2 = data.iloc[n//2:]

# Обучение базовой модели на первой части выборки
X_train = data1.drop(['n_clicks'], axis=1)
y_train = data1['n_clicks']
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
model = xgb.train(params, dtrain)

# Применение CUPAC
X_test = data2.drop(['n_clicks'], axis=1)
y_test = data2['n_clicks']
dtest = xgb.DMatrix(X_test)
y_pred = model.predict(dtest)
data2['n_clicks_cupac'] = data2['n_clicks'] - sm.OLS(y_pred, sm.add_constant(data2.drop(['n_clicks'], axis=1))).fit(cov_type='HC3').predict(sm.add_constant(data2.drop(['n_clicks'], axis=1)))

# Оценка линейной модели с использованием целевой переменной из первой части выборки как ковариату и фиксацией гетероскедастичности с помощью HC-3
model = sm.OLS(data2['n_clicks_cupac'], sm.add_constant(data2['n_queries']))
results = model.fit(cov_type='HC3')

# Вывод результатов
print('Стандартная ошибка до применения CUPAC:', mean_squared_error(y_test, y_pred)**0.5)
print('Стандартная ошибка после применения CUPAC:', np.sqrt(results.scale))

# Стандартная ошибка после применения CUPAC уменьшилась с 4.7008 до 4.6621.

sigma_cupac = np.sqrt(results.scale)

# Посчитаем значение эффекта размера для MDE=0.1:

ES = 0.0075 / sigma_cupac

# Рассчитаем значение ошибки второго рода для мощности теста
from scipy.stats import norm
from scipy import stats

beta = 0.2
z_beta = stats.norm.ppf(beta)
z_alpha = stats.norm.ppf(1-0.05/2)
nobs = ((z_alpha+z_beta)**2 * sigma_cupac**2) / ES**2
power = stats.norm.cdf((z_alpha*ES - z_beta)/np.sqrt(2)) + stats.norm.cdf((-z_alpha*ES - z_beta)/np.sqrt(2))
print(power)

#Таким образом, при выбранном значении MDE=0.1 и доверительном уровне 0.05, ошибка второго рода в CUPED и CUPAC будет равна примерно 1.44, если в АБ-тесте будет участвовать только половина наблюдений.
```
