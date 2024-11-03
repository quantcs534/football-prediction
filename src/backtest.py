import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



class FutPredict:
    def __init__(self, history_size: int = 6):
        """
        maç tahmin sistemi
        args:
            history_size: Tahmin için kullanılacak geçmiş maç sayısı
        """
        self.history_size = history_size
        self.team_mapping = {}
        self.team_names = {}
        self.feature_columns = None

    def create_team_mappings(self, data: pd.DataFrame):
        """
        takım isimlerini ve ID'lerini eşleştir
        """
        unique_home_teams = data['HomeTeam'].unique()
        unique_away_teams = data['AwayTeam'].unique()
        unique_teams = np.unique(np.concatenate([unique_home_teams, unique_away_teams]))


        self.team_mapping = {}
        self.team_names = {}

        for idx, team in enumerate(sorted(unique_teams), 1):  #sıralı ve 1den başlayan IDler
            self.team_mapping[team] = idx
            self.team_names[idx] = team

        return self.team_mapping, self.team_names

    def _calculate_match_statistics(self,
                                  match: pd.Series,
                                  is_home: bool,
                                  match_length: int = 90) -> Dict:
        """
        tek bir maç için istatistikleri hesapla
        """
        prefix = 'home_team_' if is_home else 'away_team_'
        team = match['HomeTeam'] if is_home else match['AwayTeam']
        opponent = match['AwayTeam'] if is_home else match['HomeTeam']

        shots = match[f'{prefix}shot']
        shots_on_target = match[f'{prefix}shoton']

        stats = {
            'date': pd.to_datetime(match['Date']),
            'team_name': team,
            'op_team_name': opponent,
            'teamid': self.team_mapping[team],
            'op_teamid': self.team_mapping[opponent],
            'is_home': 1 if is_home else 0,

            #(dakika başına normalize)
            'shots': shots / match_length,
            'shots_on_target': shots_on_target / match_length,
            'passes': shots / match_length,
            'bad_passes': (shots - shots_on_target) / match_length,
            'pass_ratio': shots_on_target / shots if shots > 0 else 0,
            'corners': match[f'{prefix}corner'] / match_length,
            'fouls': match[f'{prefix}foulcommit'] / match_length,
            'cards': match[f'{prefix}yellowcard'] + match[f'{prefix}redcard'],
            'goals': match[f'{prefix}goal'],

            'pass_70': (shots_on_target * 0.7) / match_length,  # Son 1/3'lük alan
            'pass_80': (shots_on_target * 0.4) / match_length,  # Son 1/5'lik alan

            'match_length': match_length,
            'season': pd.to_datetime(match['Date']).year,
            'month': pd.to_datetime(match['Date']).month,
        }

        stats['expected_goals'] = self._calculate_expected_goals(
            shots=shots,
            shots_on_target=shots_on_target,
            goals=stats['goals']
        )

        return stats

    def _calculate_expected_goals(self,
                                shots: int,
                                shots_on_target: int,
                                goals: int,
                                base_conversion: float = 0.3) -> float:
        """
        (xG) hesaplama
        """
        if shots == 0:
            return 0.0

        shot_accuracy = shots_on_target / shots
        goal_conversion = goals / shots_on_target if shots_on_target > 0 else 0

        position_quality = 0.7
        chance_quality = 0.8

        # xG hesaplama
        base_xg = shots * shot_accuracy * base_conversion
        quality_adjusted_xg = base_xg * position_quality * chance_quality

        historical_weight = 0.3
        current_weight = 0.7

        weighted_xg = (quality_adjusted_xg * current_weight +
                      (goals * base_conversion) * historical_weight)

        return weighted_xg

    def prepare_match_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        tüm maçlar için veri hazırlama
        """
        self.create_team_mappings(data)

        data = data.sort_values('Date').reset_index(drop=True)

        all_match_stats = []

        for idx, match in data.iterrows():
            match_id = idx + 1
            match_length = 90  #standart maç uzunluğu

            home_stats = {
                'matchid': match_id,
                'date': pd.to_datetime(match['Date']),
                'team_name': match['HomeTeam'],
                'op_team_name': match['AwayTeam'],
                'teamid': self.team_mapping[match['HomeTeam']],
                'op_teamid': self.team_mapping[match['AwayTeam']],
                'is_home': 1,

                'shots': match['home_team_shot'] / match_length,
                'shots_on_target': match['home_team_shoton'] / match_length,
                'passes': match['home_team_shot'] / match_length,
                'bad_passes': (match['home_team_shot'] - match['home_team_shoton']) / match_length,
                'corners': match['home_team_corner'] / match_length,
                'fouls': match['home_team_foulcommit'] / match_length,
                'cards': match['home_team_yellowcard'] + match['home_team_redcard'],
                'goals': match['home_team_goal'],

                'pass_70': (match['home_team_shoton'] * 0.7) / match_length,
                'pass_80': (match['home_team_shoton'] * 0.4) / match_length,
            }

            away_stats = {
                'matchid': match_id,
                'date': pd.to_datetime(match['Date']),
                'team_name': match['AwayTeam'],
                'op_team_name': match['HomeTeam'],
                'teamid': self.team_mapping[match['AwayTeam']],
                'op_teamid': self.team_mapping[match['HomeTeam']],
                'is_home': 0,

                'shots': match['away_team_shot'] / match_length,
                'shots_on_target': match['away_team_shoton'] / match_length,
                'passes': match['away_team_shot'] / match_length,
                'bad_passes': (match['away_team_shot'] - match['away_team_shoton']) / match_length,
                'corners': match['away_team_corner'] / match_length,
                'fouls': match['away_team_foulcommit'] / match_length,
                'cards': match['away_team_yellowcard'] + match['away_team_redcard'],
                'goals': match['away_team_goal'],

                'pass_70': (match['away_team_shoton'] * 0.7) / match_length,
                'pass_80': (match['away_team_shoton'] * 0.4) / match_length,
            }

            home_stats['expected_goals'] = self._calculate_expected_goals(
                shots=match['home_team_shot'],
                shots_on_target=match['home_team_shoton'],
                goals=match['home_team_goal']
            )

            away_stats['expected_goals'] = self._calculate_expected_goals(
                shots=match['away_team_shot'],
                shots_on_target=match['away_team_shoton'],
                goals=match['away_team_goal']
            )

            home_stats.update({
                f'op_{key}': away_stats[key]
                for key in ['shots', 'shots_on_target', 'passes', 'bad_passes',
                          'corners', 'fouls', 'cards', 'goals', 'pass_70',
                          'pass_80', 'expected_goals']
            })

            away_stats.update({
                f'op_{key}': home_stats[key]
                for key in ['shots', 'shots_on_target', 'passes', 'bad_passes',
                          'corners', 'fouls', 'cards', 'goals', 'pass_70',
                          'pass_80', 'expected_goals']
            })

            home_stats.update(self._calculate_ratios(home_stats, away_stats))
            away_stats.update(self._calculate_ratios(away_stats, home_stats))

            home_stats['points'] = self._calculate_points(
                home_stats['goals'], away_stats['goals'])
            away_stats['points'] = self._calculate_points(
                away_stats['goals'], home_stats['goals'])

            all_match_stats.extend([home_stats, away_stats])

        return pd.DataFrame(all_match_stats)

    def _calculate_ratios(self, team_stats: Dict, opponent_stats: Dict) -> Dict:
        """
        İki takım arasındaki oran istatistiklerini hesaplar
        """
        ratios = {}

        base_stats = ['shots', 'passes', 'goals']
        for stat in base_stats:
            if opponent_stats[stat] > 0:
                ratios[f'{stat}_ratio'] = team_stats[stat] / opponent_stats[stat]
            else:
                ratios[f'{stat}_ratio'] = team_stats[stat] if team_stats[stat] > 0 else 1.0

        if team_stats['shots'] > 0:
            ratios['shot_conversion'] = team_stats['goals'] / team_stats['shots']
            ratios['shot_accuracy'] = team_stats['shots_on_target'] / team_stats['shots']
        else:
            ratios['shot_conversion'] = 0
            ratios['shot_accuracy'] = 0

        return ratios
    def _calculate_points(self, team_goals: int, opponent_goals: int) -> int:
        """Maç sonucu puanı hesaplama"""
        if team_goals > opponent_goals:
            return 3
        elif team_goals == opponent_goals:
            return 1
        return 0

    def calculate_historical_features(self,
                                    match_data: pd.DataFrame,
                                    cutoff_date: datetime = None) -> pd.DataFrame:
        """
        her takım için tarihsel özellikler
        """
        if cutoff_date is None:
            cutoff_date = match_data['date'].max()

        features = []

        for team_id in match_data['teamid'].unique():
            team_matches = match_data[
                (match_data['teamid'] == team_id) &
                (match_data['date'] < cutoff_date)
            ].sort_values('date')

            if len(team_matches) < self.history_size:
                continue

            for i in range(self.history_size, len(team_matches)):
                current_match = team_matches.iloc[i]
                match_date = current_match['date']

                #son N maçı al
                last_n_matches = team_matches[
                    (team_matches['date'] < match_date)
                ].tail(self.history_size)

                #zaman bazlı ağırlıklar
                weights = self._calculate_time_weights(
                    last_n_matches['date'],
                    match_date
                )

                #ağırlıklı özellikler
                historical_features = self._calculate_weighted_features(
                    last_n_matches,
                    weights,
                    current_match
                )

                features.append(historical_features)

        return pd.DataFrame(features)

    def _calculate_time_weights(self,
                              dates: pd.Series,
                              target_date: datetime) -> np.ndarray:
        """
        tarihe göre ağırlık hesaplama
        daha yakın maçlar daha önemli
        """
        days_diff = (target_date - dates).dt.days
        weights = 1 / (days_diff + 1)**0.5  #kare kök ile yumuşatma*
        return weights / weights.sum()

    def _calculate_weighted_features(self, matches: pd.DataFrame,
                               weights: np.ndarray,
                               current_match: pd.Series) -> Dict:
        """
        ağırlıklı özellikler hesaplama
        """
        stats = {
            'matchid': current_match['matchid'],
            'date': current_match['date'],
            'teamid': current_match['teamid'],
            'op_teamid': current_match['op_teamid'],
            'is_home': current_match['is_home'],
            'team_name': current_match['team_name'],
            'op_team_name': current_match['op_team_name'],

            #hedef değişkenler
            'points': current_match['points'],
            'goals': current_match['goals'],
            'op_goals': current_match['op_goals'],
        }

        base_stats = ['shots', 'shots_on_target', 'passes', 'corners',
                    'fouls', 'cards', 'expected_goals', 'pass_70', 'pass_80']

        for stat in base_stats:
            if stat in matches.columns:
                stats[f'avg_{stat}'] = np.average(matches[stat], weights=weights)
            if f'op_{stat}' in matches.columns:
                stats[f'op_avg_{stat}'] = np.average(matches[f'op_{stat}'], weights=weights)

        ratio_stats = ['shots_ratio', 'goals_ratio', 'passes_ratio',
                      'shot_conversion', 'shot_accuracy']

        for stat in ratio_stats:
            if stat in matches.columns:
                stats[f'avg_{stat}'] = np.average(matches[stat], weights=weights)

        stats['recent_form'] = self._calculate_form_trend(matches['points'], weights)
        stats['goal_trend'] = self._calculate_form_trend(matches['goals'], weights)
        stats['defense_trend'] = self._calculate_form_trend(matches['op_goals'].multiply(-1), weights)

        return stats


    def _calculate_form_trend(self, values: pd.Series, weights: np.ndarray) -> float:
        """
        form trendi hesaplama
        ağırlıklı lineer regresyon eğimi
        """
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1, w=weights)
        return z[0]  # Eğim

    def prepare_model_features(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        model için feature matrix hazırlar
        """
        model_data = features_df[features_df['points'] != 1].copy()

        available_features = [
            'is_home',
            'avg_shots', 'avg_shots_on_target',
            'avg_passes', 'avg_corners', 'avg_fouls',
            'avg_cards', 'avg_expected_goals',
            'avg_pass_70', 'avg_pass_80',
            'op_avg_shots', 'op_avg_passes',
            'op_avg_expected_goals',
            'avg_shots_ratio', 'avg_goals_ratio',
            'avg_passes_ratio', 'avg_shot_conversion',
            'avg_shot_accuracy', 'recent_form',
            'goal_trend', 'defense_trend'
        ]

        feature_columns = [f for f in available_features if f in model_data.columns]
        self.feature_columns = feature_columns

        X = model_data[feature_columns]
        y = (model_data['points'] == 3).astype(int)

        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[LogisticRegression, Dict]:
        """
        model eğitimi ve özellik önem analizi
        """
        model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            C=0.1,  #L2 regularizasyon
            max_iter=1000
        )

        model.fit(X, y)

        importance = dict(zip(X.columns, model.coef_[0]))
        feature_importance = {
            'positive_features': [],
            'negative_features': [],
            'dropped_features': []
        }

        for feature, coef in importance.items():
            if abs(coef) < 0.001:
                feature_importance['dropped_features'].append((feature, 0))
            elif coef > 0:
                feature_importance['positive_features'].append(
                    (feature, np.exp(coef) - 1))
            else:
                feature_importance['negative_features'].append(
                    (feature, np.exp(coef) - 1))

        return model, feature_importance

    def predict_match(self,
                     model: LogisticRegression,
                     home_team: str,
                     away_team: str,
                     match_date: datetime,
                     historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        gelecek maç için tahmin
        """
        home_features = self._get_team_latest_features(
            historical_data,
            self.team_mapping[home_team],
            match_date,
            is_home=True
        )

        away_features = self._get_team_latest_features(
            historical_data,
            self.team_mapping[away_team],
            match_date,
            is_home=False
        )

        prediction_features = pd.DataFrame([home_features, away_features])[self.feature_columns]

        win_probs = model.predict_proba(prediction_features)

        results = pd.DataFrame({
            'match_date': [match_date, match_date],
            'team': [home_team, away_team],
            'opponent': [away_team, home_team],
            'is_home': [1, 0],
            'win_probability': win_probs[:, 1] * 100,
            'loss_probability': win_probs[:, 0] * 100,
            'expected_goals': [home_features['avg_expected_goals'],
                             away_features['avg_expected_goals']],
            'form_trend': [home_features['recent_form'],
                          away_features['recent_form']],
            'defensive_strength': [-home_features['defense_trend'],
                                 -away_features['defense_trend']]
        })

        return results

    def _get_team_latest_features(self,
                                data: pd.DataFrame,
                                team_id: int,
                                match_date: datetime,
                                is_home: bool) -> Dict:
        """
        takımın en son durumunu hesaplar
        """
        team_matches = data[
            (data['teamid'] == team_id) &
            (data['date'] < match_date)
        ].sort_values('date')

        if len(team_matches) < self.history_size:
            return self._get_default_features(is_home)

        last_n_matches = team_matches.tail(self.history_size)
        weights = self._calculate_time_weights(last_n_matches['date'], match_date)

        features = self._calculate_weighted_features(
            last_n_matches,
            weights,
            last_n_matches.iloc[-1]
        )

        features['is_home'] = int(is_home)

        return features

    def _get_default_features(self, is_home: bool) -> Dict:
        """
        yeni veya az maçı olan takımlar için varsayılan özellikler
        """
        defaults = {
            'is_home': int(is_home),
            'avg_shots': 10.0,
            'avg_shots_on_target': 4.0,
            'avg_passes': 300.0,
            'avg_corners': 5.0,
            'avg_fouls': 10.0,
            'avg_cards': 2.0,
            'avg_expected_goals': 1.2,
            'avg_pass_70': 100.0,
            'avg_pass_80': 50.0,
            'op_avg_shots': 10.0,
            'op_avg_passes': 300.0,
            'op_avg_expected_goals': 1.2,
            'avg_shots_ratio': 1.0,
            'avg_goals_ratio': 1.0,
            'avg_passes_ratio': 1.0,
            'avg_shot_conversion': 0.1,
            'avg_shot_accuracy': 0.4,
            'recent_form': 0.0,
            'goal_trend': 0.0,
            'defense_trend': 0.0
        }

        if hasattr(self, 'feature_columns'):
            return {k: v for k, v in defaults.items() if k in self.feature_columns}
        return defaults

    def evaluate_predictions(self,
                           predictions: pd.DataFrame,
                           actuals: pd.DataFrame) -> Dict:
        """
        tahmin performansını değerlendirir
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        from sklearn.metrics import roc_auc_score

        y_true = (actuals['points'] == 3).astype(int)
        y_pred = (predictions['win_probability'] > 50).astype(int)
        y_prob = predictions['win_probability'] / 100

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_prob),
            'baseline': y_true.mean(),
            'lift': accuracy_score(y_true, y_pred) / y_true.mean()
        }

        return metrics

    def prepare_future_fixtures(self, fixture_data: pd.DataFrame) -> pd.DataFrame:
        """
        gelecek haftanın maçlarını tahmin için hazırlar
        """
        fixture_data = fixture_data.sort_values('Date')

        future_matches = []
        for idx, match in fixture_data.iterrows():
            future_matches.append({
                'match_date': pd.to_datetime(match['Date']),
                'home_team': match['HomeTeam'],
                'away_team': match['AwayTeam']
            })

        return pd.DataFrame(future_matches)

    def predict_fixture_matches(self,
                              completed_matches: pd.DataFrame,
                              fixture_matches: pd.DataFrame) -> pd.DataFrame:
        """
        fikstürdeki gelecek maçlar için tahmin yapar
        """
        match_data = self.prepare_match_data(completed_matches)

        cutoff_date = completed_matches['Date'].max()

        historical_features = self.calculate_historical_features(match_data, cutoff_date)

        X, y = self.prepare_model_features(historical_features)
        model, _ = self.train_model(X, y)

        predictions = []

        for _, fixture in fixture_matches.iterrows():
            prediction = self.predict_future_match(
                model=model,
                home_team=fixture['home_team'],
                away_team=fixture['away_team'],
                prediction_date=fixture['match_date'],
                historical_data=match_data
            )

            match_prediction = {
                'match_date': fixture['match_date'],
                'home_team': fixture['home_team'],
                'away_team': fixture['away_team'],
                'home_win_prob': prediction.iloc[0]['win_probability'],
                'draw_prob': 20,
                'away_win_prob': prediction.iloc[1]['win_probability'],
                'predicted_home_goals': prediction.iloc[0]['expected_goals'],
                'predicted_away_goals': prediction.iloc[1]['expected_goals'],
                'home_form': prediction.iloc[0]['recent_form'],
                'away_form': prediction.iloc[1]['recent_form']
            }

            probs = [match_prediction['home_win_prob'],
                    match_prediction['draw_prob'],
                    match_prediction['away_win_prob']]
            outcomes = ['HOME', 'DRAW', 'AWAY']
            match_prediction['predicted_outcome'] = outcomes[np.argmax(probs)]

            predictions.append(match_prediction)

        return pd.DataFrame(predictions)


    def predict_future_match(self,
                        model: LogisticRegression,
                        home_team: str,
                        away_team: str,
                        prediction_date: datetime,
                        historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        sonucu bilinmeyen gelecek bir maç için tahmin yapar
        """
        past_data = historical_data[historical_data['date'] < prediction_date]

        home_features = self._prepare_team_features(
            team_name=home_team,
            is_home=1,  #ev sahibi
            prediction_date=prediction_date,
            historical_data=past_data
        )

        away_features = self._prepare_team_features(
            team_name=away_team,
            is_home=0,  #deplasman
            prediction_date=prediction_date,
            historical_data=past_data
        )

        prediction_features = pd.DataFrame([home_features, away_features])

        X_pred = prediction_features[self.feature_columns]

        win_probabilities = model.predict_proba(X_pred)

        results = pd.DataFrame({
            'match_date': [prediction_date, prediction_date],
            'team': [home_team, away_team],
            'opponent': [away_team, home_team],
            'is_home': [1, 0],
            'win_probability': win_probabilities[:, 1] * 100,
            'draw_probability': 20,
            'loss_probability': win_probabilities[:, 0] * 100,
            'expected_goals': [
                home_features.get('avg_expected_goals', 1.2),
                away_features.get('avg_expected_goals', 1.0)
            ],
            'recent_form': [
                home_features.get('recent_form', 0),
                away_features.get('recent_form', 0)
            ],
            'team_strength': [
                home_features.get('avg_goals', 0),
                away_features.get('avg_goals', 0)
            ]
        })

        return results

    def _prepare_team_features(self,
                              team_name: str,
                              is_home: int,
                              prediction_date: datetime,
                              historical_data: pd.DataFrame) -> Dict:
        """
        bir takım için tahmin özelliklerini hazırlar
        """
        # Takımın geçmiş maçlarını al
        team_id = self.team_mapping[team_name]
        team_matches = historical_data[
            historical_data['teamid'] == team_id
        ].sort_values('date')

        last_n_matches = team_matches.tail(self.history_size)

        if len(last_n_matches) < self.history_size:
            #yeterli maç yoksa varsayılan değerler kullan
            return self._get_default_features(is_home)

        #zaman bazlı ağırlıklar
        weights = self._calculate_time_weights(
            last_n_matches['date'],
            prediction_date
        )

        features = self._calculate_weighted_features(
            last_n_matches,
            weights,
            last_n_matches.iloc[-1]
        )

        #is_home değerini güncelle
        features['is_home'] = is_home

        return features
    


class CumulativeBacktest:
    def __init__(self, predictor: FutPredict, verbose: bool = True):
        self.predictor = predictor
        self.verbose = verbose
        self.predictions: List[Dict] = []
        self.metrics: Dict[str, List[float]] = {
            'accuracy': [],
            'avg_win_prob': [],
            'matches_count': []
        }

    def run_backtest(self, data: pd.DataFrame, start_date: datetime) -> pd.DataFrame:
        """backtest çalıştır"""
        data = data.sort_values('Date').copy()

        unique_dates = sorted(data[data['Date'] > start_date]['Date'].unique())

        if self.verbose:
            print(f"Başlangıç tarihi: {start_date}")
            print(f"Tahmin edilecek gün sayısı: {len(unique_dates)}")

        all_predictions = []

        for prediction_date in unique_dates:
            train_data = data[
                (data['Date'] >= data['Date'].min()) &
                (data['Date'] < prediction_date)
            ].copy()

            test_matches = data[data['Date'] == prediction_date].copy()

            if self.verbose:
                print(f"\nTahmin tarihi: {prediction_date.date()}")
                print(f"Eğitim veri boyutu: {len(train_data)} maç")
                print(f"Tahmin edilecek: {len(test_matches)} maç")

            if len(test_matches) > 0:
                daily_predictions = self._predict_day(
                    train_data=train_data,
                    test_matches=test_matches,
                    prediction_date=prediction_date
                )

                all_predictions.append(daily_predictions)

                if self.verbose:
                    self._print_daily_metrics(prediction_date, daily_predictions)

        if all_predictions:
            final_predictions = pd.concat(all_predictions, ignore_index=True)
            return final_predictions
        else:
            return pd.DataFrame()

    def _predict_day(self, train_data: pd.DataFrame,
                test_matches: pd.DataFrame,
                prediction_date: datetime) -> pd.DataFrame:
        """belirli bir gün için tahminleri yap"""
        match_data = self.predictor.prepare_match_data(train_data)
        historical_features = self.predictor.calculate_historical_features(
            match_data,
            prediction_date
        )

        X, y = self.predictor.prepare_model_features(historical_features)
        model, _ = self.predictor.train_model(X, y)

        predictions_list = []

        for _, match in test_matches.iterrows():
            try:
                if match['HomeTeam'] not in self.predictor.team_mapping or \
                  match['AwayTeam'] not in self.predictor.team_mapping:
                    if self.verbose:
                        print(f"Atlandı: {match['HomeTeam']} vs {match['AwayTeam']} - Yeni takım(lar)")
                    continue

                prediction = self.predictor.predict_match(
                    model=model,
                    home_team=match['HomeTeam'],
                    away_team=match['AwayTeam'],
                    match_date=match['Date'],
                    historical_data=match_data
                )

                for i in range(2):
                    home = bool(i == 0)
                    pred_dict = {
                        'match_date': prediction.iloc[i]['match_date'],
                        'team': prediction.iloc[i]['team'],
                        'opponent': prediction.iloc[i]['opponent'],
                        'is_home': 1 if home else 0,
                        'win_probability': prediction.iloc[i]['win_probability'],
                        'loss_probability': prediction.iloc[i]['loss_probability'],
                        'expected_goals': prediction.iloc[i]['expected_goals'],
                        'form_trend': 0.199651 if home else 0.599946, 
                        'defensive_strength': 0.164224 if home else -0.149952 
                    }
                    predictions_list.append(pred_dict)

            except KeyError as e:
                if self.verbose:
                    print(f"Hata: {match['HomeTeam']} vs {match['AwayTeam']} - {str(e)}")
                continue

        if not predictions_list:
            return pd.DataFrame() 

        return pd.DataFrame(predictions_list)

    def _print_daily_metrics(self, date: datetime, predictions: pd.DataFrame):
        """günlük metrikleri yazdır"""
        if len(predictions) == 0:
            print(f"\n=== {date.date()} Tahminler ===")
            print("Tahmin yapılacak uygun maç bulunamadı")
            return

        print(f"\n=== {date.date()} Tahminler ===")
        print(f"Maç sayısı: {len(predictions)//2}")
        print("Ortalama kazanma olasılığı: {:.2f}%".format(predictions['win_probability'].mean()))

def run_backtest(data: pd.DataFrame, start_date: datetime) -> pd.DataFrame:
    """backtest'i çalıştır"""
    predictor = FutPredict(history_size=6)
    backtest = CumulativeBacktest(predictor=predictor)

    predictions = backtest.run_backtest(
        data=data,
        start_date=start_date
    )

    return predictions