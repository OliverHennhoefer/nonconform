"""Dataset registry with metadata for all available datasets."""

from dataclasses import dataclass


@dataclass
class DatasetInfo:
    """Metadata for a dataset."""

    name: str
    description: str
    filename: str
    samples: int
    features: int
    anomaly_rate: float


DATASET_REGISTRY = {
    "breast": DatasetInfo(
        name="breast",
        description=(
            "Breast Cancer Wisconsin (Diagnostic) dataset. Contains features "
            "computed from a digitized image of a fine needle aspirate (FNA) "
            "of a breast mass. They describe characteristics of the cell "
            "nuclei present in the image."
        ),
        filename="breast_w.npz",
        samples=683,
        features=9,
        anomaly_rate=0.3499,
    ),
    "fraud": DatasetInfo(
        name="fraud",
        description=(
            "Credit card fraud detection dataset. Contains transactions made "
            "by European cardholders. It presents transactions that occurred "
            "in two days, with features that are numerical input variables, "
            "the result of a PCA transformation."
        ),
        filename="fraud.npz",
        samples=284807,
        features=29,
        anomaly_rate=0.0017,
    ),
    "ionosphere": DatasetInfo(
        name="ionosphere",
        description=(
            "Ionosphere dataset. Radar data collected by a system in Goose "
            "Bay, Labrador. The targets were free electrons in the "
            "ionosphere. Good radar returns show evidence of structure in "
            "the ionosphere."
        ),
        filename="ionosphere.npz",
        samples=351,
        features=33,
        anomaly_rate=0.3590,
    ),
    "mammography": DatasetInfo(
        name="mammography",
        description=(
            "Mammography dataset. Used for detecting breast cancer based on "
            "mammographic findings. Contains features related to BI-RADS "
            "assessment, age, shape, margin, and density."
        ),
        filename="mammography.npz",
        samples=11183,
        features=6,
        anomaly_rate=0.0232,
    ),
    "musk": DatasetInfo(
        name="musk",
        description=(
            "Musk (Version 2) dataset. Describes a set of 102 molecules of "
            "which 39 are judged by human experts to be musks and the "
            "remaining 63 molecules are judged to be non-musks. The 166 "
            "features describe the three-dimensional conformation of the "
            "molecules."
        ),
        filename="musk.npz",
        samples=3062,
        features=166,
        anomaly_rate=0.0317,
    ),
    "shuttle": DatasetInfo(
        name="shuttle",
        description=(
            "Shuttle dataset. Contains data from a NASA space shuttle mission "
            "concerning the position of radiators in the shuttle. The dataset "
            "helps identify normal and anomalous states."
        ),
        filename="shuttle.npz",
        samples=49097,
        features=9,
        anomaly_rate=0.0715,
    ),
    "thyroid": DatasetInfo(
        name="thyroid",
        description=(
            "Thyroid Disease (ann-thyroid) dataset. Used for diagnosing "
            "thyroid conditions based on patient attributes and test results. "
            "Helps identify normal and abnormal thyroid conditions."
        ),
        filename="thyroid.npz",
        samples=3772,
        features=6,
        anomaly_rate=0.0247,
    ),
    "wbc": DatasetInfo(
        name="wbc",
        description=(
            "Wisconsin Breast Cancer (Original) dataset. Contains features "
            "derived from clinical observations of breast cancer, such as "
            "clump thickness, cell size uniformity, etc. Distinct from the "
            "breast dataset which is the Diagnostic version."
        ),
        filename="wbc.npz",
        samples=223,
        features=9,
        anomaly_rate=0.0448,
    ),
    "annthyroid": DatasetInfo(
        name="annthyroid",
        description=(
            "Ann-Thyroid dataset. A dataset for thyroid disease detection "
            "from the UCI repository, containing patient attributes for "
            "diagnosing normal and abnormal thyroid conditions."
        ),
        filename="annthyroid.npz",
        samples=7200,
        features=6,
        anomaly_rate=0.0742,
    ),
    "backdoor": DatasetInfo(
        name="backdoor",
        description=(
            "Backdoor dataset. Network intrusion detection dataset "
            "containing network traffic patterns for identifying backdoor "
            "attacks and normal network behavior."
        ),
        filename="backdoor.npz",
        samples=95329,
        features=196,
        anomaly_rate=0.0244,
    ),
    "cardio": DatasetInfo(
        name="cardio",
        description=(
            "Cardiovascular disease dataset. Contains patient health "
            "indicators and measurements for predicting cardiovascular "
            "disease risk and identifying healthy individuals."
        ),
        filename="cardio.npz",
        samples=1831,
        features=21,
        anomaly_rate=0.0961,
    ),
    "cover": DatasetInfo(
        name="cover",
        description=(
            "Forest cover type dataset. Predicts forest cover type from "
            "cartographic variables. Contains wilderness area and soil type "
            "information for classification."
        ),
        filename="cover.npz",
        samples=286048,
        features=10,
        anomaly_rate=0.0096,
    ),
    "donors": DatasetInfo(
        name="donors",
        description=(
            "Blood donors dataset. Contains information about blood donors "
            "including their donation history, demographics, and patterns "
            "for predicting future donations."
        ),
        filename="donors.npz",
        samples=619326,
        features=10,
        anomaly_rate=0.0593,
    ),
    "glass": DatasetInfo(
        name="glass",
        description=(
            "Glass identification dataset. Contains measurements of glass "
            "samples including refractive index and chemical composition "
            "for identifying different types of glass."
        ),
        filename="glass.npz",
        samples=214,
        features=7,
        anomaly_rate=0.0421,
    ),
    "hepatitis": DatasetInfo(
        name="hepatitis",
        description=(
            "Hepatitis dataset. Medical dataset containing patient "
            "information and test results for hepatitis diagnosis and "
            "outcome prediction."
        ),
        filename="hepatitis.npz",
        samples=80,
        features=19,
        anomaly_rate=0.1625,
    ),
    "http": DatasetInfo(
        name="http",
        description=(
            "HTTP network traffic dataset. Contains HTTP request/response "
            "patterns for network intrusion detection and identifying "
            "normal vs. abnormal web traffic."
        ),
        filename="http.npz",
        samples=567498,
        features=3,
        anomaly_rate=0.0039,
    ),
    "letter": DatasetInfo(
        name="letter",
        description=(
            "Letter recognition dataset. Contains features extracted from "
            "handwritten letters for character recognition and "
            "identification of different letter classes."
        ),
        filename="letter.npz",
        samples=1600,
        features=32,
        anomaly_rate=0.0625,
    ),
    "lymphography": DatasetInfo(
        name="lymphography",
        description=(
            "Lymphography dataset. Medical dataset containing lymphatic "
            "system imaging data for diagnosing lymphatic diseases and "
            "normal conditions."
        ),
        filename="lymphography.npz",
        samples=148,
        features=18,
        anomaly_rate=0.0405,
    ),
    "magic_gamma": DatasetInfo(
        name="magic_gamma",
        description=(
            "MAGIC Gamma Telescope dataset. Contains measurements from the "
            "MAGIC gamma-ray telescope for distinguishing gamma rays from "
            "hadron background noise."
        ),
        filename="magic_gamma.npz",
        samples=19020,
        features=10,
        anomaly_rate=0.3516,
    ),
    "mnist": DatasetInfo(
        name="mnist",
        description=(
            "MNIST handwritten digits dataset. Contains images of "
            "handwritten digits (0-9) commonly used for image "
            "classification and anomaly detection tasks."
        ),
        filename="mnist.npz",
        samples=7603,
        features=100,
        anomaly_rate=0.0921,
    ),
    "optdigits": DatasetInfo(
        name="optdigits",
        description=(
            "Optical recognition of handwritten digits dataset. Contains "
            "features extracted from handwritten digit images for digit "
            "classification tasks."
        ),
        filename="optdigits.npz",
        samples=5216,
        features=64,
        anomaly_rate=0.0288,
    ),
    "pageblocks": DatasetInfo(
        name="pageblocks",
        description=(
            "Page blocks classification dataset. Contains features "
            "extracted from document layout analysis for classifying "
            "different types of page blocks."
        ),
        filename="pageBlocks.npz",
        samples=5393,
        features=10,
        anomaly_rate=0.0946,
    ),
    "pendigits": DatasetInfo(
        name="pendigits",
        description=(
            "Pen-based recognition of handwritten digits dataset. "
            "Contains coordinate information from pen trajectories for "
            "handwritten digit recognition."
        ),
        filename="pendigits.npz",
        samples=6870,
        features=16,
        anomaly_rate=0.0227,
    ),
    "satimage2": DatasetInfo(
        name="satimage2",
        description=(
            "Satellite image dataset (version 2). Contains multi-spectral "
            "satellite imagery data for land cover classification and "
            "anomaly detection."
        ),
        filename="satimage2.npz",
        samples=5803,
        features=36,
        anomaly_rate=0.0122,
    ),
    "smtp": DatasetInfo(
        name="smtp",
        description=(
            "SMTP network traffic dataset. Contains SMTP protocol traffic "
            "patterns for email security analysis and spam/intrusion "
            "detection."
        ),
        filename="smtp.npz",
        samples=95156,
        features=3,
        anomaly_rate=0.0003,
    ),
    "stamps": DatasetInfo(
        name="stamps",
        description=(
            "Stamps dataset. Contains features extracted from stamp images "
            "for classification and authenticity verification tasks."
        ),
        filename="stamps.npz",
        samples=340,
        features=9,
        anomaly_rate=0.0912,
    ),
    "vowels": DatasetInfo(
        name="vowels",
        description=(
            "Vowel recognition dataset. Contains acoustic features for "
            "vowel sound classification and speech recognition tasks."
        ),
        filename="vowels.npz",
        samples=1456,
        features=12,
        anomaly_rate=0.0343,
    ),
    "wine": DatasetInfo(
        name="wine",
        description=(
            "Wine quality dataset. Contains chemical analysis measurements "
            "of wines for quality assessment and classification tasks."
        ),
        filename="wine.npz",
        samples=129,
        features=13,
        anomaly_rate=0.0775,
    ),
    "yeast": DatasetInfo(
        name="yeast",
        description=(
            "Yeast dataset. Biology dataset for predicting protein "
            "localization sites in cells. Contains features describing "
            "cellular characteristics used to classify protein sequences."
        ),
        filename="yeast.npz",
        samples=1484,
        features=8,
        anomaly_rate=0.3416,
    ),
}
