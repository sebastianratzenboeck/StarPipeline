import os
import numpy as np
import astropy.units as u
from base import PipelineStep
from data import Star
from yso_model_loaders import SPUBHMI, SP__H_I, S___SMI, SP__HMI
from utils import RobustBatchLinearInterpolator



class YSO(PipelineStep):
    def __init__(self, base_dir: str, yso_models='all'):
        self._yso_model_names = ['spubhmi', 'sp--h-i', 's---smi', 'sp--hmi'] if yso_models == 'all' else yso_models
        self.base_dir = base_dir
        self.__yso_model = None
        self.init_yso_models()

    @property
    def yso_models(self):
        return self._yso_model_names

    @yso_models.setter
    def yso_models(self, new_yso_models: list):
        self.init_yso_models(new_yso_models)

    def init_yso_models(self, yso_models: list = None):
        if yso_models is None:
            yso_models = self._yso_model_names
        # Instantiating fitting models
        print('Loading YSO models, this might take a few seconds to minutes...')
        possible_models = {'spubhmi': SPUBHMI, 'sp--h-i': SP__H_I, 's---smi': S___SMI, 'sp--hmi': SP__HMI}
        self.__yso_model = {}
        for model_name in yso_models:
            if model_name in possible_models:
                self.__yso_model[model_name] = possible_models[model_name](os.path.join(self.base_dir, f'{model_name}'))

    def transform(self, data: Star) -> Star:
        # Check if the yso_model is set
        if self.__yso_model is None:
            raise ValueError("No YSO model set. Please set a YSO model using set_yso_model() method.")
        # Get the SED for the given data
        df = data.to_pandas()
        # 20...apertures, 200...wavelengths
        f_lambda_all = np.full(shape=(df.shape[0], 20, 200), fill_value=np.nan) * u.erg / u.s / u.cm**2 / u.AA
        avail_model = self._yso_model_names[0]
        wave = self.__yso_model[avail_model].wave
        # Only process models which have stellar parameters
        processed = np.asarray(df.logR.isna())
        model_used = np.asarray(['' for _ in range(df.shape[0])], dtype=object)
        # Different sources need different YSO models
        sources2model = [
            ('spubhmi', df.has_disk & df.has_envelope),
            ('s---smi', df.has_ambient_medium & ~df.has_disk),
            ('sp--h-i', ~df.has_ambient_medium & df.has_disk),
            ('sp--hmi', df.has_disk & ~df.has_envelope)
        ]
        # Get the SED for each model combination
        for model_name, condition in sources2model:
            if model_name in self.__yso_model:
                condition &= ~processed
                if condition.sum() != 0:
                    print(f'Processing {model_name} for {condition.sum()} sources')
                    # Get the SED for the model
                    f_lambda_m_i, _ = self.__yso_model[model_name].sed(df.loc[condition])
                    f_lambda_all[condition] = f_lambda_m_i
                    processed[condition] = True
                    model_used[condition] += model_name

        # Add the SED to the data dictionary
        data.flam_yso = RobustBatchLinearInterpolator(f_lambda_all, self.__yso_model[avail_model].apertures.value)
        data.wavelength_yso = wave
        data.is_single_star_without_medium = ~processed & ~df.logR.isna()
        data.model_used = model_used
        return data
