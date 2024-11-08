# Определение функции распределения масс протозвезд после слияний

## Описание проекта
Данный проект предназначен для того, чтобы смоделировать процесс слияния некоторого начального распределения N протозвезд и получить функцию итогового распределения масс после L слияний. Процесс слияния происходит по следующим правилам:

1. Слияния происходят с определенной периодичностью, и в каждом промежутке времени проходит только одно слияние.
2. У каждой протозвезды равная вероятность войти в сливающуюся пару, вне зависимости от того, испытала она слияние или нет.

В проекте используются следующие модели начального распределения масс протозвезд:

- **Модель A**: Равномерное распределение:

  $ M_i = M_0 $

- **Модель B**: Степенное распределение масс в промежутке $[M_1, M_2]$:

  $f(M) = M^{-\beta},$
  
  где значение $\beta$ находится в промежутке $[2, 3]$

- **Модель C**: Логнормальное распределение в промежутке $[M_1, M_2]$:

  $f(M) = \frac{1}{\sigma \sqrt{2 \pi}} \exp \left( -0.5 \left( \frac{x - x_0}{\sigma} \right)^2 \right), \quad x = \ln(M)$

## Использованные методы и алгоритмы
Сначала для заданного начального распределения с помощью метода Монте-Карло находится итоговое распределение масс, строится его гистограмма, а затем для функции из заданного набора моделей законов распределения по методу наименьших квадратов (МНК) подбираются параметры, при которых она наилучшим образом соответствует гистограмме. С помощью информационного критерия AIC выбирается лучшая модель.

### Модели законов распределения:
- **Нормальное распределение**:
  
  $f(x) = \frac{1}{\sqrt{2 \pi} \sigma} \exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right)$

- **Логнормальное распределение**:
  
  $f(x) = \frac{1}{\sigma \sqrt{2 \pi}} \exp \left( - \frac{(\ln x - \mu)^2}{2 \sigma^2} \right)$

- **Экспоненциальное распределение**:
  
  $f(x) = c_1 \exp (-c_2 x)$

### Критерий AIC
Критерий AIC заключается в вычислении следующего значения для каждой модели и выборе модели с наименьшим значением:

$$
AIC = 2k + n \ln \left( \frac{RSS}{n} \right)
$$

где:
- $k$ – количество параметров модели,
- $n$ – количество наблюдений,
- $RSS$ – сумма квадратов невязок.

### Результат
В качестве результата строится график, на котором присутствует диаграмма итогового распределения и график полученной модели распределения.

## Использованные библиотеки
В данном проекте использовались следующие библиотеки:
- **NumPy** – для различных математических методов
- **SciPy** – для фиттинга кривых на гистограмму
- **Multiprocessing** – для распараллеливания вычислений для больших объемов данных
- **Logging** – для гибкой настройки логирования и вывода сообщений в консоль
- **Matplotlib** – для графического представления результата
- **Collections** и **Dataclasses** – для повышения ясности кода и его гибкости

## Указания по применению
В папке `input examples` лежат примеры входных файлов с параметрами для каждой из указанных моделей. С их помощью создается входной файл `inputFile.txt`.

### Запуск программы
Программа запускается с помощью команды:

```bash
python3 main.py
