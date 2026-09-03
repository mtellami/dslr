"""Custom exception hierarchy for the DSLR project."""


class DSLRError(Exception):
    """Base class for all DSLR project errors."""


class DatasetError(DSLRError):
    """Base class for dataset-related errors."""


class DatasetNotFoundError(DatasetError):
    """Raised when an expected dataset file cannot be found."""


class DatasetFormatError(DatasetError):
    """Raised when a dataset cannot be parsed or has an invalid format."""


class MissingColumnError(DatasetError):
    """Raised when a required dataset column is missing."""


class StatisticsError(DSLRError):
    """Raised when manual statistics input is invalid."""


class DescribeError(DSLRError):
    """Raised when describe-specific processing cannot continue."""


class HistogramError(DSLRError):
    """Raised when histogram analysis or rendering cannot continue."""


class VisualizationError(DSLRError):
    """Raised when a plot cannot be rendered or written."""


class FeatureSelectionError(DSLRError):
    """Raised when the model feature-selection contract is invalid."""


class PairPlotError(VisualizationError):
    """Raised when pair-plot analysis or rendering cannot continue."""


class PlottingDependencyError(VisualizationError):
    """Raised when a required plotting dependency is unavailable."""


class PreprocessingError(DSLRError):
    """Raised when preprocessing configuration or data is invalid."""


class ZeroVarianceFeatureError(PreprocessingError):
    """Raised when a selected training feature has unusable variance."""


class PreprocessingStateError(PreprocessingError):
    """Raised when serialized preprocessing state is malformed."""


class ModelError(DSLRError):
    """Base class for model-related errors."""


class ModelFormatError(ModelError):
    """Raised when a model file has an invalid or unsupported format."""


class ModelIOError(ModelError):
    """Raised when a model file cannot be read or written safely."""


class PredictionError(ModelError):
    """Raised when prediction output cannot be produced safely."""


class ValidationError(ModelError):
    """Raised when accuracy validation cannot be completed safely."""


class LogisticRegressionError(ModelError):
    """Raised when binary logistic-regression inputs are invalid."""


class TrainingConfigurationError(LogisticRegressionError):
    """Raised when binary training configuration is invalid."""


class OptimizationError(LogisticRegressionError):
    """Raised when binary optimization produces non-finite values."""


class OneVsRestError(ModelError):
    """Raised when one-vs-rest multiclass processing fails."""


class OneVsRestStateError(OneVsRestError):
    """Raised when one-vs-rest model state is malformed."""
