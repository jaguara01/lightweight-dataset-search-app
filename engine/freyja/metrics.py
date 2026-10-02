import numpy as np
import jellyfish


class Metric():
    def __init__(self, name, active, normalize, distance_pattern):
        self.name = name
        self.active = active
        self.normalize = normalize
        self.distance_pattern = distance_pattern

class MetricProperties():
    def __init__(self):
        metrics_groups = [CardinalityMetrics(), DistributionMetrics(), CommonValuesMetrics(),
                           LengthMetrics(), WordCountMetrics(), BoundaryMetrics(), ColumnFlagsMetrics()]
        
        self.metrics_to_normalize = []
        for metric_group in metrics_groups:
            for metric in metric_group.metrics_list:
                if metric.normalize and metric.active:
                    self.metrics_to_normalize.append(metric.name)
        
        self.distance_patterns = {}
        for metric_group in metrics_groups:
            for metric in metric_group.metrics_list:
                if metric.active:
                    self.distance_patterns[metric.name] = metric.distance_pattern


class CardinalityMetrics():
    def __init__(self):
        self.cardinality = Metric('cardinality', True, True, "substraction")
        self.incompleteness = Metric('incompleteness', True, False, "substraction")
        self.uniqueness = Metric('uniqueness', True, False, "substraction")
        self.entropy = Metric('entropy', True, True, "substraction")


        self.metrics = {
            self.cardinality.name: self.cardinality.active,
            self.incompleteness.name:  self.incompleteness.active,
            self.uniqueness.name:  self.uniqueness.active,
            self.entropy.name:  self.entropy.active,
        }

        self.metrics_list = [self.cardinality, self.incompleteness, self.uniqueness, self.entropy]

    def build_query(self, column, table):
        query = f"""
            SELECT
                COUNT(DISTINCT "{column}") AS {self.cardinality.name},
                ENTROPY("{column}") AS {self.entropy.name},
                COUNT(DISTINCT "{column}") AS {self.uniqueness.name},
                COUNT(*) - COUNT("{column}") AS {self.incompleteness.name}
            FROM "{table}"
        """
        return query
    
    def process_result(self, result, count_rows):
        for key in (self.uniqueness.name, self.incompleteness.name):
            result[key] = result[key] / count_rows if count_rows > 0 else 0

        filtered_result = {k: v for k, v in result.items() if self.metrics.get(k)}
        return filtered_result
    

class DistributionMetrics():
    def __init__(self):
        self.frequency_avg = Metric('frequency_avg', True, True, "substraction")
        self.frequency_min = Metric('frequency_min', True, True, "substraction")
        self.frequency_max = Metric('frequency_max', True, True, "substraction")
        self.frequency_sd = Metric('frequency_sd', True, True, "substraction")

        self.skewness = Metric('skewness', False, True, "substraction") ##############
        self.kurtosis = Metric('kurtosis', False, True, "substraction") ##############

        self.val_pct_min = Metric('val_pct_min', True, False, "substraction")
        self.val_pct_max = Metric('val_pct_max', True, False, "substraction")
        self.val_pct_sd = Metric('val_pct_std', True, False, "substraction")
        self.constancy = Metric('constancy', True, False, "substraction")

        self.frequency_iqr = Metric('frequency_iqr', True, False, "substraction")
        self.frequency_1qo = Metric('frequency_1qo', True, False, "substraction")
        self.frequency_2qo = Metric('frequency_2qo', True, False, "substraction")
        self.frequency_3qo = Metric('frequency_3qo', True, False, "substraction")
        self.frequency_4qo = Metric('frequency_4qo', True, False, "substraction")
        self.frequency_5qo = Metric('frequency_5qo', True, False, "substraction")
        self.frequency_6qo = Metric('frequency_6qo', True, False, "substraction")
        self.frequency_7qo = Metric('frequency_7qo', True, False, "substraction")


        self.metrics = {
            self.frequency_avg.name: self.frequency_avg.active,
            self.frequency_min.name:  self.frequency_min.active,
            self.frequency_max.name:  self.frequency_max.active,
            self.frequency_sd.name:  self.frequency_sd.active,
            self.frequency_iqr.name:  self.frequency_iqr.active,

            self.skewness.name:  self.skewness.active,
            self.kurtosis.name:  self.kurtosis.active,

            self.val_pct_min.name:  self.val_pct_min.active,
            self.val_pct_max.name:  self.val_pct_max.active,
            self.val_pct_sd.name:  self.val_pct_sd.active,
            self.constancy.name:  self.constancy.active,

            self.frequency_1qo.name: self.frequency_1qo.active,
            self.frequency_2qo.name: self.frequency_2qo.active,
            self.frequency_3qo.name: self.frequency_3qo.active,
            self.frequency_4qo.name: self.frequency_4qo.active,
            self.frequency_5qo.name: self.frequency_5qo.active,
            self.frequency_6qo.name: self.frequency_6qo.active,
            self.frequency_7qo.name: self.frequency_7qo.active,
        }

        self.metrics_list = [self.frequency_avg, self.frequency_min, self.frequency_max, self.frequency_sd, self.frequency_iqr,
                             self.skewness, self.kurtosis,
                             self.val_pct_min, self.val_pct_max, self.val_pct_sd, self.constancy,
                             self.frequency_1qo, self.frequency_2qo, self.frequency_3qo, self.frequency_4qo, self.frequency_5qo,
                             self.frequency_6qo, self.frequency_7qo]

    def build_query(self, column, table):
        query = f"""
            SELECT 
                AVG(count) AS {self.frequency_avg.name},
                MIN(count) AS {self.frequency_min.name}, 
                MAX(count) AS {self.frequency_max.name},
                STDDEV_POP(count) AS {self.frequency_sd.name},
                (quantile_disc(count, 0.75) - quantile_disc(count, 0.25)) AS {self.frequency_iqr.name},
                SKEWNESS(count) AS {self.skewness.name},
                KURTOSIS(count) AS {self.kurtosis.name},
                quantile_disc(count, 0.125) AS frequency_1qo, 
                quantile_disc(count, 0.25) AS frequency_2qo,
                quantile_disc(count, 0.375) AS frequency_3qo,
                quantile_disc(count, 0.5) AS frequency_4qo,
                quantile_disc(count, 0.625) AS frequency_5qo, 
                quantile_disc(count, 0.75) AS frequency_6qo,
                quantile_disc(count, 0.875) AS frequency_7qo
            FROM (
                SELECT COUNT("{column}") AS count
                FROM "{table}"
                WHERE "{column}" IS NOT NULL
                GROUP BY "{column}"
            )
        """
        return query
    
    def process_result(self, result, count_rows):
        for key in (self.skewness.name, self.kurtosis.name):
            result[key] = 0 if result[key] is None or np.isnan(result[key]) else result[key]

        for key in (self.frequency_iqr.name,
                    self.frequency_1qo.name, self.frequency_2qo.name, self.frequency_3qo.name, self.frequency_4qo.name,
                    self.frequency_5qo.name, self.frequency_6qo.name, self.frequency_7qo.name):
            result[key] = result[key] / count_rows if count_rows > 0 else 0
            
        result[self.val_pct_min.name] = result[self.frequency_min.name] / count_rows if count_rows > 0 else 0
        result[self.val_pct_max.name] = result[self.frequency_max.name] / count_rows if count_rows > 0 else 0
        result[self.val_pct_sd.name] = result[self.frequency_sd.name] / count_rows if count_rows > 0 else 0
        result[self.constancy.name] = result[self.frequency_max.name] / count_rows if count_rows > 0 else 0
        
    
        filtered_result = {k: v for k, v in result.items() if self.metrics.get(k)}
        return filtered_result


class CommonValuesMetrics():
    def __init__(self):
        self.freq_word_containment = Metric('freq_word_containment', True, False, "containment")
        self.freq_word_soundex_containment = Metric('freq_word_soundex_containment', True, False, "containment")

        self.metrics = {
            self.freq_word_containment.name: self.freq_word_containment.active,
            self.freq_word_soundex_containment.name:  self.freq_word_soundex_containment.active,
        }

        self.metrics_list = [self.freq_word_containment, self.freq_word_soundex_containment]

    def build_query(self, column, table):
        query = f"""
            SELECT "{column}", COUNT("{column}") as count
            FROM "{table}"
            WHERE "{column}" IS NOT NULL
            GROUP BY "{column}"
            ORDER BY count DESC, "{column}" ASC
            LIMIT 10
        """
        return query
    
    def process_result(self, common_values):
        result = {}
        result[self.freq_word_containment.name] = [row[0] for row in common_values]
        result[self.freq_word_soundex_containment.name] = [
            jellyfish.soundex(str(val)) if val is not None else None 
            for val in result[self.freq_word_containment.name]
        ]

        filtered_result = {k: v for k, v in result.items() if self.metrics.get(k)}
        return filtered_result
    
class LengthMetrics():
    def __init__(self):
        self.len_max_word = Metric('len_max_word', True, True, "substraction")
        self.len_min_word = Metric('len_min_word', True, True, "substraction")
        self.len_avg_word = Metric('len_avg_word', True, True, "substraction")


        self.metrics = {
            self.len_max_word.name: self.len_max_word.active,
            self.len_min_word.name:  self.len_min_word.active,
            self.len_avg_word.name:  self.len_avg_word.active
        }

        self.metrics_list = [self.len_max_word, self.len_min_word, self.len_avg_word]

    def build_query(self, column, table):
        query = f"""
            SELECT 
                MAX(max_string_length) AS {self.len_max_word.name}, 
                MIN(min_string_length) AS {self.len_min_word.name}, 
                AVG(avg_string_length) AS {self.len_avg_word.name}
            FROM (
                SELECT 
                    list_aggregate(list_transform(str_split(CAST("{column}" AS VARCHAR), ' '), s -> LENGTH(s)), 'max') AS max_string_length,
                    list_aggregate(list_transform(str_split(CAST("{column}" AS VARCHAR), ' '), s -> LENGTH(s)), 'min') AS min_string_length,
                    list_aggregate(list_transform(str_split(CAST("{column}" AS VARCHAR), ' '), s -> LENGTH(s)), 'avg') AS avg_string_length
                FROM "{table}"
                WHERE "{column}" IS NOT NULL
            )
        """
        return query
    
    def process_result(self, result):
        filtered_result = {k: v for k, v in result.items() if self.metrics.get(k)}
        return filtered_result
    

class WordCountMetrics():
    def __init__(self):
        self.number_words = Metric('number_words', True, True, "substraction")
        self.words_cnt_max = Metric('words_cnt_max', True, True, "substraction")
        self.words_cnt_min = Metric('words_cnt_min', True, True, "substraction")
        self.words_cnt_avg = Metric('words_cnt_avg', True, True, "substraction")
        self.words_cnt_sd = Metric('words_cnt_sd', True, True, "substraction")


        self.metrics = {
            self.number_words.name: self.number_words.active,
            self.words_cnt_max.name:  self.words_cnt_max.active,
            self.words_cnt_min.name:  self.words_cnt_min.active,
            self.words_cnt_avg.name:  self.words_cnt_avg.active,
            self.words_cnt_sd.name:  self.words_cnt_sd.active
        }

        self.metrics_list = [self.number_words, self.words_cnt_max, self.words_cnt_min, self.words_cnt_avg, self.words_cnt_sd]

    def build_query(self, column, table):
        query = f"""
            SELECT 
                SUM(num_words) AS {self.number_words.name}, 
                MAX(num_words) AS {self.words_cnt_max.name}, 
                MIN(num_words) AS {self.words_cnt_min.name},
                AVG(num_words) AS {self.words_cnt_avg.name}, 
                STDDEV_POP(num_words) AS {self.words_cnt_sd.name}
            FROM (
                SELECT LENGTH(CAST("{column}" AS VARCHAR)) - LENGTH(REPLACE(CAST("{column}" AS VARCHAR), ' ', '')) + 1 AS num_words
                FROM "{table}"
                WHERE "{column}" IS NOT NULL
            )
        """
        return query
    
    def process_result(self, result):
        filtered_result = {k: v for k, v in result.items() if self.metrics.get(k)}
        return filtered_result
    

class BoundaryMetrics():
    def __init__(self):
        self.first_word = Metric('first_word', True, False, "levenshtein")
        self.last_word = Metric('last_word', True, False, "levenshtein")


        self.metrics = {
            self.first_word.name: self.first_word.active,
            self.last_word.name:  self.last_word.active
        }

        self.metrics_list = [self.first_word, self.last_word]

    def build_query(self, column, table):
        query = f"""
            SELECT 
                MIN("{column}") AS first_word, 
                MAX("{column}") AS last_word
            FROM "{table}"
            WHERE "{column}" IS NOT NULL
        """
        return query
    
    def process_result(self, result):
        filtered_result = {k: v for k, v in result.items() if self.metrics.get(k)}
        return filtered_result
    

class ColumnFlagsMetrics():
    def __init__(self):
        self.is_empty = Metric('is_empty', True, False, "substraction")
        self.is_binary = Metric('is_binary', True, False, "substraction")


        self.metrics = {
            self.is_empty.name: self.is_empty.active,
            self.is_binary.name:  self.is_binary.active
        }

        self.metrics_list = [self.is_empty, self.is_binary]

    def build_query(self, column, table):
        query = f"""
            SELECT COUNT(DISTINCT "{column}") AS count_distinct
            FROM "{table}"
            WHERE "{column}" IS NOT NULL
        """
        return query
    
    def process_result(self, count_distinct):
        result = {}
        result[self.is_empty.name] = 1 if count_distinct[0] == 0 else 0
        result[self.is_binary.name] = 1 if count_distinct[0] == 2 else 0

        filtered_result = {k: v for k, v in result.items() if self.metrics.get(k)}
        return filtered_result