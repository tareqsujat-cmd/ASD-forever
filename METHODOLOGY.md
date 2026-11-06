# Contributions

This study makes four contributions. The first two concern how autism is expressed in
functional connectivity in ABIDE-I, and the last two concern analytical choices that
shape measured classification performance on this dataset. Under the designations of
Section 8.1, the first two rest on exploratory analyses, the third on a primary
descriptive comparison, and the fourth on a secondary comparison.

1. **Connectivity-based detection is weaker in autistic participants with higher
   cognitive ability.** After site, age, and mean framewise displacement were
   regressed from both variables, classifier scores declined with full-scale IQ within
   the autistic group ($\rho=-0.18$, $p=0.0004$, $n=376$) but not within the
   non-autistic group ($\rho=0.05$, $p=0.32$, $n=435$), and against all 468
   non-autistic participants as a fixed comparison group, pooled ten-fold AUROC fell
   from 0.873 below full-scale IQ 85 to 0.719 at 115 or above, with the same decline
   under leave-one-site-out validation. Matching the groups on full-scale IQ within
   site leaves discrimination highest in the lowest IQ quartile but removes the
   decline across the upper three, so part of the fixed-group gradient reflects the IQ
   difference between groups; the within-group association nonetheless shows that
   measured performance on ABIDE-I depends on the cognitive composition of the
   autistic sample and that connectivity-based detection is least reliable in
   cognitively able autistic individuals.

   *Analyses: Sections 1.5, 6.9, and 8.4.*

2. **Somatomotor hypoconnectivity in autism strengthens under motion matching.** In
   exploratory analyses without global signal regression, the number of within- and
   between-system pairs with reduced connectivity at Benjamini–Hochberg $q<0.05$ rose
   from 10 of 36 in the 863-participant group-contrast cohort to 26 of 36 in 256
   within-site pairs matched for mean framewise displacement, and the mean
   standardised effect of the 21 somatomotor regions strengthened from $-0.215$ in the
   863-participant group-contrast cohort to $-0.249$ in the 280 pairs retained by the
   wider regional caliper, with all 21 remaining significant. Because autistic
   participants moved more and head motion was positively associated with whole-brain
   mean connectivity ($\rho=0.51$ across participants and $0.45$ within site),
   residual motion would on average raise connectivity in autistic participants,
   opposite to the observed reduction; the direction agrees with the somatomotor
   involvement reported by Chen et al. (2015) and the somatomotor reductions reported
   by Holiga et al. (2019) and Ilioska et al. (2023) in samples that include ABIDE
   data, and here it is obtained in within-site pairs matched for head motion.

   *Analyses: Sections 1.3, 1.5, 1.6, 7.1–7.4, 8.1, and 8.3.*

3. **Feature selection outside the training partition inflates performance, and the
   inflation depends on feature granularity.** Selecting 2,000 of 19,900 static
   tangent connections by F test with all diagnostic labels before cross-validation
   raised fold-mean AUROC from 0.756 without selection to 0.987 and accuracy from
   0.692 to 0.948 on identical folds, whereas ranking 10 to 100 of 196 regions with
   all labels rather than within each training fold (863 participants) changed
   participant-pooled AUROC by a mean absolute 0.006, with inconsistent sign. Without
   any such step, the three-representation ensemble reached a participant-pooled
   accuracy of 0.706 (95% Wilson CI 0.678–0.733, $n=1{,}032$) under the
   quality-control definition used by published ABIDE-I benchmarks, comparable to the
   0.700, 0.703, and 0.708 reported by Heinsfeld et al. (2018), Eslami et al. (2019),
   and Almuqhim and Saeed (2021), so accuracy at that level does not require
   out-of-partition selection.

   *Analyses: Sections 1.4, 4.4, 4.5, 6.7, and 6.10.*

4. **Linear classifiers outperform fixed-configuration tree ensembles on static
   tangent-space connectivity.** With identical input features and folds, the five
   linear classifiers reached participant-pooled AUROCs of 0.735–0.757, against
   0.611–0.624 for random forest, XGBoost, and CatBoost, and nested
   $\ell_2$-regularised logistic regression exceeded each tree ensemble by at least
   0.13 (unadjusted DeLong $p\le2.4\times10^{-12}$). The comparison, made on 871
   participants and 19,900 features, favours linear models; because the tree ensembles
   used fixed configurations, it characterises these configurations rather than the
   best attainable member of each family, although the fixed-configuration ridge,
   linear support-vector, and discriminant classifiers also exceeded all three.

   *Analyses: Sections 4.4, 6.3, 8.1, and 8.2.*

Other analyses, including transfer to unseen sites, probability calibration, and the
symptom associations of Section 6.8, are reported as descriptive results, and
comparisons that did not improve discrimination are reported in their respective
subsections as limits on what these data support.

# Methods

## 1. Study design and cohort

### 1.1 Study design

Autism spectrum disorder (ASD) was defined by the diagnostic label recorded in the
Autism Brain Imaging Data Exchange I (ABIDE-I). Clinical diagnosis combines direct
observation, developmental history, and professional judgement. Resting-state
functional magnetic resonance imaging (rs-fMRI) was used to characterise large-scale
functional connectivity in autistic participants and non-autistic participants.

Participants with `DX_GROUP = 1` were coded as autistic, and participants with
`DX_GROUP = 2` were coded as non-autistic. The primary estimand was discrimination
between these labels in the ABIDE-I research cohort. The principal descriptive
endpoints were participant-pooled out-of-fold area under the receiver operating
characteristic curve (AUROC) under pooled ten-fold and leave-one-site-out validation.

The primary prediction model was an equal-weight ensemble of three connectivity
representations. Five additional objectives were evaluated. The first four correspond
to the contributions stated above: discrimination across levels of cognitive ability;
adjusted differences among functional systems, with and without matching for head
motion; the effect of feature-selection scope and granularity; and the effect of
classifier family under a fixed representation and validation procedure. The fifth,
the association of classifier scores with symptom measures within autistic
participants and across both diagnostic groups, was analysed descriptively.

The analysis proceeded in eight stages:

1. The participant cohort was defined from the ABIDE-I quality ratings.
2. Regional Configurable Pipeline for the Analysis of Connectomes (C-PAC) time series
   were loaded from the selected preprocessing derivative.
3. Correlation, static tangent-space, and windowed tangent-space features were
   constructed.
4. Feature scaling and regularised logistic regression were fitted within each outer
   training partition.
5. The three member probabilities were averaged.
6. Pooled ten-fold and leave-one-site-out validation were performed.
7. Planned method comparisons and adjusted group-contrast analyses were conducted.
8. Fold-level metrics and out-of-fold predictions were stored for statistical
   comparison.

### 1.2 Data source and preprocessing

Imaging and phenotypic records were obtained from ABIDE-I, a multi-site aggregation of
research acquisitions (Di Martino et al., 2014). The ABIDE-I release contained 1,112
participants with phenotypic records and quality ratings for anatomical and functional
images. The Preprocessed Connectomes Project supplied regional time series processed
through a common pipeline (Craddock et al., 2013). Site-related variation in scanners,
acquisition protocols, scan lengths, participant characteristics, and image quality
formed part of the multi-site sample (Abraham et al., 2017).

Imaging inputs were obtained from the C-PAC `filt_noglobal` derivative. The download
specified the C-PAC pipeline, enabled band-pass filtering, and disabled global signal
regression. The downloaded regional time series were used directly in the analyses.

The documented C-PAC workflow applied slice-timing adjustment, two-stage motion
realignment, skull stripping, intensity normalisation, and anatomical registration.
Nuisance regression included a 24-parameter motion model, five component-based noise
modelling (CompCor) components from white-matter and cerebrospinal-fluid signals, and
linear and quadratic trends. Band-pass filtering from 0.01 to 0.10 Hz followed
nuisance regression. The `filt_noglobal` strategy excluded the global mean signal from
the nuisance model (Craddock et al., 2013).

### 1.3 Primary strict cohort

The primary strict cohort was defined from all available image-quality ratings. A
participant was retained when:

- rater 1 marked the scan as acceptable;
- rater 2 marked both anatomical and functional images as acceptable or borderline;
  and
- rater 3 marked both anatomical and functional images as acceptable.

The strict cohort contained 871 participants: 403 autistic participants and 468
non-autistic participants. The cohort contained 727 male and 144 female participants.
Ages ranged from 6.5 to 58 years, with a mean of 16.9 years and a median of 14.8
years.

The other 241 records lacked at least one required passing rating. Release annotations
included motion, skull stripping, anatomical coverage, preprocessing, image noise,
missing modalities, and signal dropout. The annotation categories could overlap.

The primary analyses loaded the metadata generated for the strict cohort. Cohort
membership was fixed during preprocessing and verified against the phenotypic
identifiers before analysis.

Table 1 summarises participant characteristics by diagnostic group. Age and the three
IQ scores were compared between groups by Welch's t-test; mean framewise displacement
by Welch's t-test and, because its distribution was right-skewed, also by the
Mann–Whitney U test; scan length by the Mann–Whitney U test; and sex by the $\chi^2$
test with Yates' continuity correction. The dependence of discrimination on cognitive
ability was examined in Section 6.9, and the sensitivity of the group-difference
analyses to head motion in the motion-matched analyses of Section 7.3.

**Table 1.** Participant characteristics in the strict cohort (871 participants; 20
site labels from 17 acquisition sites). Values are mean (SD) unless stated; the number
of participants with an available value is given in brackets where it is below the
group total.

| Characteristic | Autistic (n = 403) | Non-autistic (n = 468) | p |
|---|---|---|---|
| Age, years | 17.07 (7.96) | 16.84 (7.24) | 0.66 |
| Female, n (%) | 54 (13.4) | 90 (19.2) | 0.027 |
| Full-scale IQ | 105.79 (17.06) [376] | 111.09 (12.08) [435] | $6.2\times10^{-7}$ |
| Verbal IQ | 104.89 (18.10) [342] | 111.57 (13.07) [379] | $2.8\times10^{-8}$ |
| Performance IQ | 105.69 (17.26) [344] | 107.91 (13.05) [391] | 0.052 |
| Mean framewise displacement (Section 1.5), mm | 0.126 (0.143) | 0.094 (0.082) | $8.2\times10^{-5}$ |
| Mean framewise displacement (Section 1.5), mm, median [IQR] | 0.088 [0.057, 0.144] | 0.068 [0.045, 0.109] | $8.9\times10^{-7}$ |
| Volumes per scan, median [range] | 176 [78, 316] | 176 [78, 316] | 0.77 |

p-values are from Welch's t-test for rows summarised by mean (SD), the Mann–Whitney U
test for rows summarised by median, and the $\chi^2$ test with Yates' continuity
correction for sex.

### 1.4 Wider quality-control cohort

The wider quality-control cohort was used for a cohort-definition sensitivity analysis
and for comparison with published reports on ABIDE-I. Eligibility required an
acceptable functional-image rating from the third rater. The phenotypic table
contained 1,035 eligible participants, and regional time series were available for
1,032 of those participants. The same eligible count of 1,035 participants was
analysed by Heinsfeld et al. (2018), Eslami et al. (2019), and Almuqhim and Saeed
(2021).

### 1.5 Phenotypic variables

Age, sex, full-scale intelligence quotient (IQ), verbal IQ, performance IQ,
handedness, and acquisition site were linked to participant identifiers. The primary
ensemble used imaging features only. A secondary phenotype model used age, sex, the
three IQ measures, and handedness. Acquisition site was used for partitioning,
nuisance adjustment, and sensitivity analyses.

Head motion was summarised by mean framewise displacement, taken from the
`func_mean_fd` quality metric distributed with the preprocessed release. The Quality
Assessment Protocol of the Preprocessed Connectomes Project computes this metric as
Jenkinson's mean framewise displacement: the root-mean-square displacement between
consecutive volumes, derived from the rigid-body realignment transforms, with
rotations converted to displacement over a sphere of 80 mm radius (Jenkinson et al.,
2002). Values are in millimetres.

Missing phenotypic values are either coded $-9999$ or left blank in the release; both
were treated as missing before any analysis. Missing covariate values were
median-imputed from the outer training partition in the phenotype model (Section 6.5)
and for age in the ComBat model (Section 6.6), in which a missing sex value would have
been assigned to the male category (sex was coded 0 for male and 1 for female); no
participant in the strict cohort lacked age or sex, so neither rule was applied. They
were median-imputed from the analysis cohort in the nuisance models of the
region-level selection comparison (Section 6.7) and of the group-difference analyses
(Section 7.1), and before motion matching (Section 7.3). In the symptom-association
(Section 6.8), cognitive-subgroup (Section 6.9), instrument-concordance (Section
6.13), and component–symptom (Section 6.16) analyses, and in Table 1, participants
with a missing value for a variable were excluded from the analyses involving that
variable, and no imputation was performed.

### 1.6 Atlas and signal quality

The primary atlas was the Craddock 200-region functional parcellation (CC200; Craddock
et al., 2012). Prediction analyses in the strict cohort used all 200 regions and
therefore produced 19,900 off-diagonal connectivity features.

A separate signal-quality procedure was used for group-difference analyses. A region
was excluded when its temporal standard deviation equalled zero in more than 1% of the
1,032 available CC200 time-series files from the wider quality-control cohort. Four
regions met this criterion. Participants in the strict cohort with a constant signal
in any retained region were then excluded from group-difference analyses. The
resulting group-contrast cohort contained 863 participants and 196 regions,
corresponding to 19,110 unique connections.

Each CC200 parcel was linked to an anatomical label and a functional system. Yeo's
seven cortical systems were resampled to the CC200 grid by nearest-neighbour
interpolation (Yeo et al., 2011). A parcel was assigned to the system with the largest
voxel overlap when that overlap exceeded 0.20. Remaining parcels were assigned to a
subcortical/cerebellar category. The assignment produced eight system classes and 36
unordered within- and between-system pairs.

### 1.7 Scan length

Scans in the strict cohort contained 78 to 316 volumes. Scan length varied by site.
Dynamic-connectivity analyses therefore used the same number of windows for every
participant. Window length and temporal change were expressed in volumes because the
analysis files lacked a harmonised repetition-time field.

## 2. Connectivity representations

### 2.1 Functional-connectivity features

An rs-fMRI scan records blood-oxygen-level-dependent signals over time. Each atlas
region was represented by one time series. Functional connectivity was defined as the
statistical dependence between pairs of regional time series. With 200 CC200 regions,
each strict-cohort participant had 19,900 unique pairwise features.

Three representations were used in the primary ensemble: shrinkage correlation,
whole-scan tangent connectivity, and mean windowed tangent connectivity. Each
representation was fitted and scaled separately.

### 2.2 Shrinkage-correlation features

For each participant, a Ledoit–Wolf covariance estimate was converted to a correlation
matrix. The strict upper triangle was vectorised, and the diagonal was discarded.
Correlation features were generated within the same outer-fold procedure as the other
representations.

### 2.3 Whole-scan covariance estimation

Let $X_i \in \mathbb{R}^{T_i \times P}$ denote the regional time series for
participant $i$, where $T_i$ is the number of volumes and $P$ is the number of
regions. Whole-scan covariance was estimated with Ledoit–Wolf shrinkage (Ledoit &
Wolf, 2004):

$$
\widehat{\Sigma}_i
=
(1-\rho_i)\widehat{S}_i
+
\rho_i \widehat{\mu}_i I_P,
\qquad
\widehat{\mu}_i
=
\frac{\operatorname{tr}(\widehat{S}_i)}{P}.
\tag{1}
$$

In Equation (1), $\widehat{S}_i$ is the sample covariance, $I_P$ is the identity
matrix, and $\rho_i$ was selected by the Ledoit–Wolf estimator. Shrinkage was used
because the number of regions was large relative to the number of volumes.

### 2.4 Tangent-space projection

Symmetric positive-definite covariance matrices lie on a curved manifold. The tangent
space is locally Euclidean at its reference matrix. For each outer fold, the reference
was the affine-invariant Riemannian mean of the training covariances (Varoquaux et
al., 2010; Dadi et al., 2019):

$$
\overline{\Sigma}_{\mathcal{T}}
=
\arg\min_{R \succ 0}
\sum_{i \in \mathcal{T}}
\left\|
\log\!\left(R^{-1/2}\widehat{\Sigma}_iR^{-1/2}\right)
\right\|_F^2.
\tag{2}
$$

In Equation (2), $\mathcal{T}$ denotes the training indices. The Riemannian mean was
estimated by a fixed-point iteration with at most ten updates. Matrix logarithms and
inverse square roots were computed from symmetric eigendecompositions. Eigenvalues
were clamped at $10^{-6}$ for numerical stability.

Training and held-out covariances were projected with the training reference:

$$
S_i
=
\log\!\left(
\overline{\Sigma}_{\mathcal{T}}^{-1/2}
\widehat{\Sigma}_i
\overline{\Sigma}_{\mathcal{T}}^{-1/2}
\right).
\tag{3}
$$

The projected matrix in Equation (3) was vectorised by retaining its off-diagonal
upper triangle:

$$
x_i
=
\operatorname{vecu}(S_i)
\in
\mathbb{R}^{P(P-1)/2}.
\tag{4}
$$

For CC200, the vector in Equation (4) contained 19,900 features. The reference matrix
was estimated from the training partition and applied unchanged to held-out
participants.

### 2.5 Additional isotropic shrinkage

The primary static tangent member used the Ledoit–Wolf estimate in Equation (1), which
corresponded to an additional shrinkage value of zero. Planned comparisons applied a
second shrinkage step:

$$
\widehat{\Sigma}_i(r)
=
(1-r)\widehat{\Sigma}_i
+
r\frac{\operatorname{tr}(\widehat{\Sigma}_i)}{P}I_P,
\qquad 0 \leq r < 1.
\tag{5}
$$

Equation (5) was used with $r=0.15$ to match the additional shrinkage applied to the
windowed representation. A nested sensitivity analysis selected $r$ from
$\{0,0.10,0.20,0.30,0.40,0.50\}$ within each outer training partition. Classifier
regularisation was reselected for every candidate value.

### 2.6 Windowed tangent-space representation

The mean windowed tangent representation formed the third member of the primary
ensemble. Dispersion, velocity, and alternative block combinations were secondary
analyses. Window-level covariance matrices were used to represent dynamic connectivity
(Hutchison et al., 2013; Allen et al., 2014; Leonardi & Van De Ville, 2015; Preti et
al., 2017).

Every participant contributed $K=6$ windows of $L=50$ volumes. For a scan with $T_i$
volumes, the nominal stride and start of window $k$ were:

$$
\Delta_i
=
\frac{T_i-L}{K-1},
\qquad
s_{ik}
=
\operatorname{round}(k\Delta_i),
\qquad
k=0,\ldots,K-1.
\tag{6}
$$

Equation (6) fixed the number of windows across participants. Shorter scans produced
greater overlap, whereas the longest scans could contain small gaps between windows.
Window starts were rounded to the nearest integer index.

Each window covariance was estimated with Ledoit–Wolf shrinkage and additional
isotropic shrinkage of $r=0.15$. Each training participant contributed the arithmetic
mean of six window covariances. The Riemannian mean of the participant-level matrices
defined the dynamic training reference. Diagonal jitter was increased adaptively when
required for window-level eigendecomposition.

The values $K=6$ and $L=50$ were used because every scan contained at least 78
volumes, so that each window spanned 50 volumes. The value $r=0.15$ was used to
stabilise the matrix logarithms of window covariances.

After tangent projection, participant $i$ was represented by
$\xi_i(1),\ldots,\xi_i(K)$. Three summaries were computed elementwise:

$$
m_i
=
\frac{1}{K}\sum_{k=1}^{K}\xi_i(k),
\tag{7}
$$

$$
d_i
=
\left[
\frac{1}{K}
\sum_{k=1}^{K}
\left(\xi_i(k)-m_i\right)^2
\right]^{1/2},
\tag{8}
$$

$$
v_i
=
\frac{1}{(K-1)\Delta_i}
\sum_{k=1}^{K-1}
\left|\xi_i(k+1)-\xi_i(k)\right|.
\tag{9}
$$

The mean in Equation (7) represented the average location of the window trajectory.
The dispersion in Equation (8) used the population denominator $K$ and represented
within-scan variability. The velocity in Equation (9) used the nominal stride from
Equation (6) and represented absolute change per volume.

## 3. Prediction model

### 3.1 Feature scaling

Each feature block was standardised with the mean and standard deviation estimated
from the outer training partition:

$$
z_{ij}
=
\frac{x_{ij}-\mu_{j,\mathcal{T}}}
{\sigma_{j,\mathcal{T}}}.
\tag{10}
$$

The transformation in Equation (10) was fitted on the training partition and applied
unchanged to the held-out partition. Separate scalers were fitted for the correlation,
static tangent, and dynamic tangent members.

### 3.2 Regularised logistic regression

Each primary member used logistic regression with an $\ell_2$ penalty. The predicted
probability for participant $i$ was:

$$
p_i
=
\sigma(w^\top z_i+b),
\qquad
\sigma(a)
=
\frac{1}{1+\exp(-a)}.
\tag{11}
$$

The model in Equation (11) was fitted in scikit-learn (Pedregosa et al., 2011) with
the `liblinear` solver and a maximum of 2,000 iterations. The inverse regularisation
parameter $C$ was selected from $\{0.001,0.01,0.1,1.0\}$ by three-fold stratified
cross-validation in the outer training partition. Selection maximised AUROC. A
separate $C$ was selected for each representation, and class weights were uniform.

For the primary ensemble, the tangent reference and scaler were fitted once on the
complete outer training partition. The transformed training matrix was reused across
the three inner splits used to select $C$. The outer held-out partition was separated
from the fitted transformations and model-selection procedure.

### 3.3 Probability ensemble

The primary model averaged probabilities from whole-scan correlation, whole-scan
tangent connectivity, and mean windowed tangent connectivity. For member probabilities
$p_i^{(1)},p_i^{(2)},p_i^{(3)}$, the ensemble probability was:

$$
p_i^{\mathrm{ens}}
=
\frac{1}{3}
\sum_{m=1}^{3}p_i^{(m)}.
\tag{12}
$$

The members in Equation (12) received equal weight. Ensemble membership and weighting
were fixed across outer folds. Equal weighting avoided estimation of additional fusion
parameters from the training sample. The default class threshold was 0.5.

### 3.4 Secondary rank fusion

As a secondary analysis, each held-out score was converted to its position in the
corresponding distribution of in-sample training scores. Ties used a right-sided
insertion rule. The three resulting scores were averaged. The rank score ranged from
zero to one and was analysed as an ordering score rather than a probability. Each
participant was scored independently of other held-out participants.

## 4. Validation protocol

### 4.1 Pooled ten-fold validation

The primary strict cohort was partitioned by stratified ten-fold cross-validation with
shuffling and random seed 42. Each participant appeared in one outer test fold. The
pooled protocol estimated performance for held-out participants drawn from the
combined ABIDE-I site mixture.

### 4.2 Leave-one-site-out validation

The primary strict cohort was also evaluated by leave-one-site-out validation. Each of
the 20 recorded site labels was held out once. Participants from the held-out site
formed the test fold, and participants from the other sites formed the training
partition. The inner search for $C$ used stratified participant folds within the outer
training data.

### 4.3 Fold-level fitting

Each outer fold used the following sequence:

1. Training and held-out participant indices were defined.
2. The tangent reference was fitted to the training covariances.
3. Training and held-out covariances were transformed with the training reference.
4. Feature scaling was fitted to the training features.
5. $C$ was selected by inner cross-validation on the transformed training matrix.
6. The selected classifier was refitted to the complete outer training partition.
7. Probabilities were predicted for the held-out partition.
8. The three member probabilities were combined.

Per-participant covariance estimation depended only on that participant's time series
and was cached before fold-specific fitting. Shared references, scalers, feature
selectors, and classifier parameters were estimated at the training level specified
for each analysis.

### 4.4 Performance measures and aggregation

Discrimination was summarised by AUROC and area under the precision–recall curve.
Threshold-dependent measures comprised accuracy, balanced accuracy, sensitivity,
specificity, F1 score (the harmonic mean of precision and recall), and Matthews
correlation coefficient at a threshold of 0.5.

Participant-pooled metrics were calculated after concatenating all out-of-fold
predictions. Unweighted outer-fold means and standard deviations assigned equal weight
to each fold. The two summaries were retained because site sizes differed in
leave-one-site-out validation. Summary tables of planned comparisons stored unweighted
fold means for AUROC and accuracy. Calibration, paired-prediction tests, and principal
interval estimates used concatenated out-of-fold predictions.

### 4.5 Interval construction

AUROC intervals were calculated from the DeLong variance. Accuracy, sensitivity, and
specificity used Wilson intervals (Wilson, 1927). The intervals conditioned on the
realised out-of-fold predictions and omitted variation from repeating the complete
model-development procedure.

Site-specific AUROCs from leave-one-site-out validation in the primary strict cohort
were summarised by a DerSimonian–Laird random-effects model (DerSimonian & Laird,
1986). The units were the 20 site labels used for leave-one-site-out validation
(Section 4.2); site labels containing one diagnostic class were excluded, leaving $J$
site labels. Site label $j$ was weighted by $1/(V_j+\hat{\tau}^2)$, where $V_j$ is the
DeLong variance of its AUROC and $\hat{\tau}^2$ the estimated between-site variance.
Heterogeneity was summarised by Cochran's $Q$ on $J-1$ degrees of freedom and by
$I^2=\max\{0,(Q-(J-1))/Q\}$, the proportion of total variation attributable to
between-site differences. The 95% prediction interval for the AUROC at an acquisition
site not represented in the analysis was:

$$
\hat{\mu}
\pm
t_{0.975,\,J-2}
\sqrt{\hat{\tau}^2+\operatorname{SE}(\hat{\mu})^2}.
\tag{13}
$$

In Equation (13), $\hat{\mu}$ is the random-effects mean AUROC,
$\operatorname{SE}(\hat{\mu})$ its standard error, and $t_{0.975,\,J-2}$ the 0.975
quantile of Student's t distribution on $J-2$ degrees of freedom.

## 5. Probability assessment and selective prediction

### 5.1 Secondary probability assessment

The native logistic-regression probabilities from Equation (12) were assessed with ten
equal-width bins. Expected calibration error (ECE) compared the mean predicted
probability that a participant was autistic with the observed proportion of autistic
participants in each bin (Naeini et al., 2015; Guo et al., 2017):

$$
\operatorname{ECE}
=
\sum_{b=1}^{10}
\frac{|\mathcal{B}_b|}{n}
\left|
\frac{1}{|\mathcal{B}_b|}
\sum_{i\in\mathcal{B}_b}p_i^{\mathrm{ens}}
-
\frac{1}{|\mathcal{B}_b|}
\sum_{i\in\mathcal{B}_b}y_i
\right|.
\tag{14}
$$

In Equation (14), $y_i=1$ denoted an autistic participant. Empty bins contributed no
term. Reliability diagrams displayed the same bin-level quantities under both
validation protocols. ECE assessed positive-class probability calibration.

### 5.2 Exploratory risk–coverage analysis

The exploratory risk–coverage analysis defined prediction confidence as:

$$
c_i
=
\max\left(p_i^{\mathrm{ens}},1-p_i^{\mathrm{ens}}\right).
\tag{15}
$$

Participants were ordered by the confidence in Equation (15). At coverage levels from
10% to 100%, the number selected, accuracy, AUROC, and selective risk were recorded.
Selective risk was the fraction of incorrect decisions among selected participants
(Geifman & El-Yaniv, 2017).

Each requested coverage retained the highest-confidence observations in the complete
out-of-fold prediction set. AUROC was undefined when a retained subset contained one
diagnostic class.

### 5.3 Exploratory finite-sample threshold analysis

The exploratory threshold analysis selected a confidence threshold from labelled
calibration data. Candidate thresholds were the distinct observed confidence values,
and a participant was selected when $c_i\geq\theta$. For candidate $\theta$,
$k(\theta)$ was the number selected and $e(\theta)$ was the number of errors. The
one-sided Clopper–Pearson upper bound (Clopper & Pearson, 1934) was:

$$
U(\theta;\delta)
=
\operatorname{BetaInv}
\left(
1-\delta;
e(\theta)+1,
k(\theta)-e(\theta)
\right).
\tag{16}
$$

In Equation (16), $\operatorname{BetaInv}$ denotes the beta-distribution quantile
function. Candidate thresholds were evaluated with Equation (16), from the most
restrictive to the most permissive. The fixed-sequence search stopped at the first
threshold whose upper bound exceeded target risk $\alpha$. Candidates selecting fewer
than ten participants were skipped (Angelopoulos et al., 2021; Angelopoulos & Bates,
2023). The last passing candidate was retained. The selected threshold was set to
1.01, with coverage of zero, when the first eligible candidate exceeded the target.
The upper bound was set to one when every selected case was an error.

The ensemble analysis used $\alpha=0.25$, $\delta=0.10$, and a minimum selected sample
of ten. The values represented a 25% target error and a 90% upper confidence bound.
Marginal thresholding used one threshold for the calibration sample. Site-conditional
thresholding used one threshold per site. The marginal threshold was used when a
site's calibration subset contained fewer than ten participants.

Threshold performance was evaluated over 200 random, unstratified half-splits. One
half selected the threshold, and the other half measured realised coverage and risk.
Repetitions with finite selective risk contributed to the summary. Reported summaries
comprised mean, median, and 90th-percentile risk, mean coverage, and the fraction of
valid repetitions meeting the target.

## 6. Planned comparisons

### 6.1 Primary descriptive comparison: static and windowed connectivity

The windowed covariance used additional shrinkage of 0.15. Static tangent connectivity
was therefore evaluated at $r=0$ and $r=0.15$. The dynamic mean was compared with the
$r=0.15$ static representation so that post-estimation shrinkage was held fixed.
Window sampling, covariance aggregation, and reference construction remained specific
to each representation.

Static shrinkage was also evaluated over
$\{0,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.65,0.80\}$. The sweep was first run over the
six values up to 0.30 and was extended to 0.80 after those results had been inspected,
in order to characterise performance at high shrinkage. The fixed-$r$ comparison was
unaffected by the extension because its value, 0.15, was set by the windowed
representation. A separate nested analysis selected shrinkage within each outer
training partition. The fixed-$r$ comparison and the data-selected shrinkage analysis
were reported separately.

### 6.2 Secondary comparison: dynamic trajectory blocks

The secondary comparison evaluated the dynamic mean alone. Concatenation combined mean
with dispersion, mean with velocity, and all three blocks under one classifier
penalty. Late fusion fitted one classifier to each constituent block and averaged the
probabilities with equal weights. Standalone metrics were retained for the dynamic
mean; dispersion and velocity contributed to the combined conditions.

### 6.3 Secondary comparison: classifier family

The secondary comparison used the same static tangent features, feature scaling, and
outer folds for all classifiers. The evaluated models were:

- nested $\ell_2$-regularised logistic regression;
- ridge classification with penalty strength 100;
- linear support-vector classification with $C=0.01$;
- shrinkage linear discriminant analysis with automatically estimated Ledoit–Wolf
  shrinkage, after 200-component principal component analysis (PCA);
- a two-stage low-rank spectral classifier of rank 20;
- random forest with 300 trees of unrestricted depth;
- XGBoost with 300 trees of maximum depth 4, learning rate 0.1, row subsampling 0.8,
  and column subsampling 0.3; and
- CatBoost with 300 trees of depth 4, learning rate 0.1, and feature subsampling 0.3.

The logistic model selected $C$ through the inner search described in Section 3.2. No
other classifier was tuned; each used the fixed configuration listed above (Breiman,
2001; Chen & Guestrin, 2016; Prokhorenkova et al., 2018). For classifiers without
`predict_proba`, the decision score was passed through a logistic function before
threshold-based metrics were calculated. Because only the logistic model was tuned,
the comparison was designated secondary, and differences between classifiers were
interpreted as differences between these configurations rather than between the best
attainable member of each model family.

### 6.4 Secondary comparison: atlas and fusion

The secondary comparison evaluated CC200, the Automated Anatomical Labeling atlas, and
the Harvard–Oxford atlas (Tzourio-Mazoyer et al., 2002; Desikan et al., 2006). The
atlases contained 200, 116, and 111 regions, respectively. Multi-atlas analyses used
the intersection of participants with valid time series for all requested atlases.

One logistic model was fitted per atlas. Atlas probabilities were combined by equal
averaging. Secondary fusion conditions compared probability averaging, stacking, and
feature concatenation.

### 6.5 Secondary comparison: phenotype model

The secondary comparison used age, sex, full-scale IQ, verbal IQ, performance IQ, and
handedness. Missing values were median-imputed from the training partition. The
variables were then standardised and entered into logistic regression. The phenotype
probability was evaluated alone and as an additional ensemble member.

### 6.6 Secondary comparison: ComBat harmonisation

The secondary comparison evaluated ComBat as a batch-adaptive procedure (Johnson et
al., 2007; Fortin et al., 2018). Site was specified as the batch variable, and age and
sex were retained as covariates. Diagnosis was excluded from the ComBat model.

Within each outer fold, training features and the complete unlabelled held-out batch
were supplied jointly to ComBat. The joint fit estimated a batch effect for a site
absent from the training data. ComBat was applied after outer-fold feature scaling. A
missing age value would have been replaced by the training median, and a missing sex
value assigned to the male category; no participant in the strict cohort lacked either
value. The prespecified fallback retained the scaled features whenever transformed
features were unavailable.

### 6.7 Primary descriptive comparison: feature-selection scope

Seven data-access conditions were evaluated with the 19,900 static tangent features
from the primary strict cohort:

1. The reference condition fitted the tangent reference, scaler, and classifier in
   each outer training fold.
2. A global-reference condition fitted the tangent reference to all participants and
   retained fold-specific scaling.
3. A global-scaling condition fitted the tangent reference and scaler to all
   participants.
4. A global-selection condition fitted the tangent reference globally, fitted scaling
   within each fold, and selected the 2,000 strongest connections with an F test using
   all labels. The approximately 10% subset served as the connection-level selection
   condition.
5. A held-out-tuning condition selected $C$ on the held-out fold.
6. A site-feature condition added one-hot site indicators to a fold-specific pipeline.
7. A combined condition applied global tangent projection, global scaling, global
   label-informed feature selection, held-out $C$ selection, and site indicators.

The seven conditions represented separate or combined changes in data access. Each
condition retained the same outer folds and primary strict cohort.

A region-level comparison used the strict-cohort group-contrast subset of 863
participants and 196 regions. Additional shrinkage was fixed at $r=0.30$. Parcels were
ranked by the average absolute group difference over their incident connections.
Models retained the top 10, 20, 30, 50, or 100 regions. The nuisance model was fitted
once to the complete analysis cohort. The outcome-dependent ranking then used either
all diagnostic labels or the labels in each outer training fold. A model containing
all retained regions was the reference condition.

### 6.8 Primary descriptive comparison: pooled and within-participant symptom associations

Out-of-fold ensemble probabilities were related to six symptom measures: Autism
Diagnostic Observation Schedule (ADOS) total and calibrated severity; the social,
verbal-communication, and restricted/repetitive-behaviour domains of the Autism
Diagnostic Interview–Revised (ADI-R); and Social Responsiveness Scale (SRS) total
(Lord et al., 1994, 2000; Constantino & Gruber, 2005). Each instrument used its
available complete cases.

Spearman correlations were calculated across all participants with an available score
and within autistic participants. The two scopes were evaluated for pooled and
leave-one-site-out predictions.

### 6.9 Exploratory comparison: cognitive and sex subgroups

The exploratory analysis linked out-of-fold predictions from the strict cohort to the
phenotypic records after verifying label and site order. Full-scale IQ was centred
within site and divided into quartiles. AUROC was estimated within each quartile.

A second analysis held all 468 non-autistic participants fixed and divided autistic
participants into full-scale IQ bands below 85, 85–99, 100–114, and 115 or above. Each
autistic band was therefore evaluated against the same non-autistic participants.
Because a fixed comparison group does not remove the group difference in full-scale
IQ, a third analysis took the autistic participants at each site in ascending order of
full-scale IQ and matched each, greedily and without replacement, to the non-autistic
participant at the same site with the closest full-scale IQ, provided that the
difference was at most five points. Autistic participants without such a match were
excluded, and AUROC was estimated within site-centred IQ quartiles of the matched
sample.

Within autistic participants and within non-autistic participants, classifier scores
were related separately to full-scale, verbal, and performance IQ. Site, age, and mean
framewise displacement were regressed from both variables before Spearman correlation.
Sex-specific AUROCs were also calculated for subgroups containing at least 20
participants and both diagnostic classes.

### 6.10 Exploratory comparison: cohort definition

The exploratory sensitivity analysis reran the static tangent classifier and the
primary ensemble (Section 3.3) on the strict cohort of 871 participants and on the
wider quality-control cohort of 1,032 participants, with the model specification held
fixed. The static tangent classifier was evaluated under pooled validation, and the
primary ensemble under both pooled and leave-one-site-out validation.

### 6.11 Exploratory comparison: scan duration and sample size

The exploratory duration analysis retained participants with at least 232 volumes and
truncated the same scans to 40, 60, 90, 120, 160, 200, and 232 volumes. Covariances
were re-estimated at every duration. The duration sweep used additional shrinkage
values of 0.15, 0.30, and 0.50.

The sample-size analysis used full-length scans and participant subsets of 100, 200,
300, 400, 550, 700, and 825. One unstratified random subset was used at each sample
size, with additional shrinkage fixed at $r=0.15$. A further analysis varied sample
size within the New York University (NYU) site and used five-fold validation. Spearman
correlation summarised the monotonic association between each resource axis and AUROC.

### 6.12 Exploratory comparison: training-cohort homogeneity

The exploratory comparison used NYU as the single-site source and a static tangent
model with additional shrinkage $r=0.30$. Each condition sampled 120 training
participants and 40 test participants without class stratification over 20 random
repetitions. The four conditions crossed one-site or multi-site training with
same-site or other-site testing. AUROC was the outcome.

### 6.13 Exploratory comparison: clinical-instrument concordance

The exploratory analysis included autistic participants with complete ADI-R algorithm
scores and an ADOS total. The ADI-R criterion required scores of at least 10 for
social interaction, 8 for verbal communication, and 3 for restricted or repetitive
behaviour. The ADOS criterion used a total score of at least 7 (Lord et al., 1994,
2000).

The four combinations of positive and negative instrument criteria and the observed
discordance rate were reported. Instrument agreement was the estimand, and agreement
beyond chance was summarised by Cohen's $\kappa$ computed from the resulting 2 × 2
table. The ABIDE-I diagnostic label remained the classification outcome.

The concordance data, together with Social Responsiveness Scale totals of non-autistic
participants, were also used to bound the accuracy attainable against the ABIDE-I
labels under the mutually contaminated label-noise model of Menon et al. (2015), in
which each observed diagnostic group is a mixture of truly autistic and truly
non-autistic participants. The noise rate $\varepsilon_1$, the proportion of
participants labelled autistic whose label was incorrect, was set to half the observed
disagreement proportion $\eta$, on the assumption that each discordant case implicates
exactly one of the two instruments. The noise rate $\varepsilon_0$, the corresponding
proportion among participants labelled non-autistic, was estimated as the proportion
of the 137 non-autistic participants with an available Social Responsiveness Scale
total whose raw total was at least 60. With $\pi_1$ and $\pi_0$ the proportions of
autistic and non-autistic participants in the strict cohort, the accuracy ceiling was
$1-(\pi_0\varepsilon_0+\pi_1\varepsilon_1)$. Observed AUROC and accuracy of the
primary ensemble were corrected with the same rates using the corresponding identities
of Menon et al. (2015): $(\mathrm{AUROC}-0.5)/(1-\varepsilon_0-\varepsilon_1)+0.5$ and
$(\mathrm{accuracy}-(\varepsilon_0+\varepsilon_1)/2)/(1-\varepsilon_0-\varepsilon_1)$.
The second identity is exact for balanced accuracy; because the two groups were not of
equal size, its application to accuracy was an approximation.

Neither rate was identified by these data. For $\varepsilon_1$, two instruments
applied to a single population do not identify their error rates whatever the cell
counts: the 2 × 2 table supplies three independent proportions against five unknowns
(two sensitivities, two specificities, and the prevalence), and identification
requires a second population with a different prevalence (Hui & Walter, 1980); in
addition, no participant was negative on both instruments. For $\varepsilon_0$, the
value 60 is conventionally a T-score threshold, whereas the release provides the raw
total. The ceiling was therefore reported as the range obtained over
$\varepsilon_1\in\{0,\eta/2,\eta\}$ and raw-total thresholds of 40, 50, 55, 60, and
70, rather than as a single value.

### 6.14 Secondary comparison: low-rank spectral approximation

The secondary analysis represented each standardised static tangent feature vector as
the upper triangle of a symmetric matrix $\widetilde{S}_i$. A dense linear coefficient
vector was mapped to a symmetric weight matrix $W$. Off-diagonal entries were divided
by two so that the matrix inner product matched the original vector dot product.

The matrix was decomposed as:

$$
W
=
U\Lambda U^\top.
\tag{17}
$$

Components in Equation (17) were ordered by the absolute value of their eigenvalues.
For rank $q$, the first $q$ components defined participant scores:

$$
z_{ik}
=
u_k^\top \widetilde{S}_i u_k,
\qquad k=1,\ldots,q.
\tag{18}
$$

The scores in Equation (18) were obtained from a dense stage-one logistic model with
$C=0.01$. A second logistic regression with $C=1.0$ was fitted to the component
scores. Ranks $\{1,2,5,10,20,30,50,100\}$ were evaluated on the same folds. A separate
spectral analysis recorded the cumulative squared-singular-value mass of the feature
matrix and the cumulative absolute eigenvalue mass of $W$.

### 6.15 Exploratory comparison: coefficient and SHAP summaries

The exploratory analysis treated model coefficients as descriptive parameters.
Group-difference interpretation was based on the residualised analyses in Section 7.
For a linear model and a training-fold background, Shapley additive explanations
(SHAP) were calculated as:

$$
\phi_{ij}
=
w_j
\left(
z_{ij}
-
\mathbb{E}_{\mathcal{T}}[z_j]
\right).
\tag{19}
$$

The analysis in Equation (19) used $C=0.01$ on the pooled outer folds. SHAP values
were computed out of fold with the corresponding training partition as the explainer
background (Lundberg & Lee, 2017). Pearson correlation between flattened calculated
and closed-form values was computed within each fold.

Mean absolute SHAP was calculated for each edge and summed into Yeo-7 system cells.
System labels were sampled from the Yeo atlas at each parcel centroid. These
assignments differed from the voxel-overlap labels used in Section 7. Unassigned edges
were omitted, and within-system edges contributed twice to diagonal cells. Three
participant-level waterfall examples and summary values were saved.

### 6.16 Exploratory comparison: component–symptom associations

The exploratory analysis fitted a rank-20 spectral classifier once to the complete
strict cohort. Rank 20 was used as a compact representative of the broader rank sweep.
The tangent reference, scaler, and classifier were fitted with the complete cohort and
its diagnostic labels. Symptom scores entered only the subsequent association
analysis.

Component scores were analysed within autistic participants. Age, sex, full-scale IQ,
and site indicators were regressed from both the component score and symptom measure.
Spearman correlation was applied to the residuals, and p-values were adjusted across
all component–instrument combinations.

Component stability was assessed by comparing components fitted in each
cross-validation training fold with components fitted to the complete strict cohort.
Absolute cosine similarity to the best-matching component summarised stability.

## 7. Exploratory group-difference analyses

### 7.1 Connection construction and nuisance adjustment

All group-difference analyses were exploratory and used the strict-cohort subset of
863 participants with 196 retained regions. Pearson correlations were Fisher
transformed and vectorised. A separate linear nuisance model was fitted for each
connection with age, sex, mean framewise displacement, and one-hot acquisition-site
indicators. Missing nuisance values were replaced by the analysis-cohort median.

Standardised group differences were calculated from the residual connection values.
For connection $e$, the effect was:

$$
d_e
=
\frac{
\overline{R}_{e,\mathrm{autistic}}
-
\overline{R}_{e,\mathrm{comparison}}
}{
\operatorname{SD}(R_e)
}.
\tag{20}
$$

In Equation (20), $\operatorname{SD}$ denotes standard deviation. The denominator was
the standard deviation of the residual across both groups. Negative values indicated
lower connectivity in autistic participants. The comparison term in Equation (20)
denoted non-autistic participants.

### 7.2 Exploratory functional-system contrasts

Connections were grouped by the functional systems assigned to their endpoints. For
each system pair containing at least 20 connections, the connection-level standardised
effects were averaged. The nuisance model and residuals were fitted once before
permutation. Each permutation recomputed the group contrast from the fixed residuals.

### 7.3 Exploratory motion-matched sensitivity analysis

Autistic participants and non-autistic participants from the same site were matched
greedily and without replacement by mean framewise displacement. Missing nuisance
values were imputed before matching with medians from the complete pre-matching
cohort.

The system-level analysis used a maximum within-pair difference of 0.01 mm. The
regional analysis used a maximum difference of 0.02 mm. Nuisance adjustment and
site-stratified permutation were repeated within each matched sample. The two calipers
produced analysis-specific matched cohorts and represented different balances between
motion proximity and retained sample size: the system-level caliper retained 256 pairs
(512 participants) and the regional caliper 280 pairs (560 participants). The group
difference in mean framewise displacement was tested by the Mann–Whitney U test before
and after matching.

To characterise the relation between head motion and overall connectivity in this
pipeline, mean connection strength was computed for each participant in the
group-contrast subset as the mean Fisher-transformed correlation over the 19,110
retained connections. Its association with mean framewise displacement was estimated
by Spearman correlation across participants and, separately, after acquisition site
had been regressed from both variables.

### 7.4 Exploratory regional analysis and internal stability

The exploratory regional analysis used a functional system supplied as an analysis
parameter; the default target was the somatomotor system. For each parcel in that
system, the effect was the mean standardised group difference across its within-system
connections. Region-level p-values were adjusted across parcels in the target system.
The regional family followed inspection of the system-level analysis.

Internal ranking stability was evaluated over 200 random split-half repetitions.
Nuisance coefficients and regional effects were refitted independently in each half.
Spearman correlation measured agreement between regional rankings. The same procedure
was applied to all within-system connections and to their top 50. One hundred randomly
drawn region sets of equal size were used as comparators. The random-region comparison
used the first 50 split-half repetitions.

## 8. Statistical analysis

### 8.1 Analysis status and multiplicity

No confirmatory hypothesis-testing family was predefined. Analyses were treated as
primary descriptive, secondary, or exploratory, as labelled in the subsection
headings. The strict-cohort ensemble supplied the primary descriptive endpoints, and
the matched-shrinkage (Section 6.1), feature-selection-scope (Section 6.7), and
pooled-versus-within-participant (Section 6.8) comparisons were the primary
descriptive comparisons. Pairwise model tests (Section 8.2), including those within
the primary descriptive comparisons, were interpreted descriptively rather than as
confirmatory tests.

Pairwise comparison p-values were reported without multiplicity adjustment and
alongside the corresponding effect size. Benjamini–Hochberg q-values across the
pairwise comparisons were reported for reference only and did not serve as a decision
criterion. Network, regional, symptom-association, and component–symptom families used
the Benjamini–Hochberg false-discovery-rate procedure (Benjamini & Hochberg, 1995).
For the symptom associations of Section 6.8, the six instruments formed one family
within each combination of validation protocol and correlation scope.

### 8.2 Paired predictive comparisons

Conditions evaluated on the same participants were compared from stored out-of-fold
predictions. DeLong's test assessed differences between correlated AUROCs (DeLong et
al., 1988). McNemar's test assessed paired disagreement between binary decisions at
the 0.5 threshold (McNemar, 1947). Both measures were reported because ranking and
thresholded decisions represented different estimands. McNemar's test was not applied
to the rank score of Section 3.4, which had no probability scale and therefore no 0.5
decision threshold.

### 8.3 Permutation analyses

Functional-system and regional analyses used 5,000 permutations of diagnostic labels
within site. The two-sided p-value included a plus-one adjustment. Here
$T_{\mathrm{obs}}$ was the observed system- or region-level effect, and $T_b^\ast$ was
the corresponding effect after permutation $b$:

$$
p
=
\frac{
1
+
\sum_{b=1}^{B}
\mathbb{1}\left(
\left|T_b^\ast\right|
\geq
\left|T_{\mathrm{obs}}\right|
\right)
}{
B+1
},
\qquad B=5000.
\tag{21}
$$

Equation (21) added one to the numerator and denominator. The smallest attainable
value at $B=5000$ was $1/5001$; values at that floor were reported as $p <
2\times10^{-4}$.

### 8.4 Correlation analyses

Symptom, cognitive-subgroup, motion–connectivity, internal-stability, and scaling
analyses used Spearman rank correlation. Each coefficient was accompanied by its
participant scope, available sample size, and adjustment procedure.

## 9. Reproducibility and scope

### 9.1 Reproducibility

Randomised analyses used seed 42 unless a repeated analysis specified a documented
offset. Top-level experiment executions created timestamped result directories and run
manifests. Each manifest recorded the arguments, seed, declared execution device,
Python and platform information, PyTorch and NumPy versions, Git commit, branch, and
working-tree state. Focused analyses wrote their outputs to the corresponding study
directory.

Fold-level summaries and pooled out-of-fold predictions were stored for later
comparison. Prediction records contained the true label, predicted probability, and
acquisition site. Phenotype linkage reconstructed the deterministic fold order and
verified labels and sites before additional variables were joined. Paired comparisons
required equal sample size and identical label order.

### 9.2 Scope

The primary analysis estimated retrospective diagnostic-group discrimination in the
strict ABIDE-I cohort of 871 participants. The wider quality-control cohort of 1,032
participants was used for the cohort-definition sensitivity analysis and for
comparison with published reports. The group-difference analyses used the
strict-cohort subset of 863 participants and 196 retained regions.

The strict cohort was predominantly male and mostly school-aged or older.
Leave-one-site-out validation estimated transfer among acquisition cohorts represented
in ABIDE-I. The ensemble probabilities reflected the class distribution of the study
cohort, and site-conditional selective thresholds required labelled calibration
participants from the relevant site.

Dynamic features were indexed in volumes. Their temporal duration therefore varied
with site-specific repetition time. Inference for windowed connectivity was restricted
to the 50-volume, six-window construction.

Group-difference estimates were conditional on the C-PAC `filt_noglobal` preprocessing
strategy, CC200 parcellation, nuisance model, and recorded mean framewise
displacement. Motion matching assessed sensitivity to the recorded motion measure.
Functional-system aggregation provided a lower-dimensional summary than individual
connections.

## References

Abraham, A., Milham, M. P., Di Martino, A., Craddock, R. C., Samaras, D., Thirion, B.,
& Varoquaux, G. (2017). Deriving reproducible biomarkers from multi-site resting-state
data: An autism-based example. *NeuroImage, 147*, 736–745.

Allen, E. A., Damaraju, E., Plis, S. M., Erhardt, E. B., Eichele, T., & Calhoun, V. D.
(2014). Tracking whole-brain connectivity dynamics in the resting state. *Cerebral
Cortex, 24*(3), 663–676.

Almuqhim, F., & Saeed, F. (2021). ASD-SAENet: A sparse autoencoder and deep-neural
network model for detecting autism spectrum disorder (ASD) using fMRI data. *Frontiers
in Computational Neuroscience, 15*, 654315.

Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction.
*Foundations and Trends in Machine Learning, 16*(4), 494–591.

Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., & Lei, L. (2021). Learn
then test: Calibrating predictive algorithms to achieve risk control.
*arXiv:2110.01052*.

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A
practical and powerful approach to multiple testing. *Journal of the Royal Statistical
Society: Series B, 57*(1), 289–300.

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.

Chen, C. P., Keown, C. L., Jahedi, A., Nair, A., Pflieger, M. E., Bailey, B. A., &
Müller, R.-A. (2015). Diagnostic classification of intrinsic functional connectivity
highlights somatosensory, default mode, and visual regions in autism. *NeuroImage:
Clinical, 8*, 238–245.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In
*Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery
and Data Mining* (pp. 785–794).

Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits
illustrated in the case of the binomial. *Biometrika, 26*(4), 404–413.

Constantino, J. N., & Gruber, C. P. (2005). *Social Responsiveness Scale*. Western
Psychological Services.

Craddock, R. C., James, G. A., Holtzheimer, P. E., Hu, X. P., & Mayberg, H. S. (2012).
A whole brain fMRI atlas generated via spatially constrained spectral clustering.
*Human Brain Mapping, 33*(8), 1914–1928.

Craddock, C., Benhajali, Y., Chu, C., Chouinard, F., Evans, A., Jakab, A., et al.
(2013). The Neuro Bureau Preprocessing Initiative: Open sharing of preprocessed
neuroimaging data and derivatives. *Frontiers in Neuroinformatics, 7*.

Dadi, K., Rahim, M., Abraham, A., Chyzhyk, D., Milham, M., Thirion, B., & Varoquaux,
G. (2019). Benchmarking functional connectome-based predictive models for
resting-state fMRI. *NeuroImage, 192*, 115–134.

DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas
under two or more correlated receiver operating characteristic curves: A nonparametric
approach. *Biometrics, 44*(3), 837–845.

DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled
Clinical Trials, 7*(3), 177–188.

Desikan, R. S., Ségonne, F., Fischl, B., Quinn, B. T., Dickerson, B. C., Blacker, C.
J., et al. (2006). An automated labeling system for subdividing the human cerebral
cortex on MRI scans into gyral based regions of interest. *NeuroImage, 31*(3),
968–980.

Di Martino, A., Yan, C.-G., Li, Q., Denio, E., Castellanos, F. X., Alaerts, K., et al.
(2014). The autism brain imaging data exchange: Towards a large-scale evaluation of
the intrinsic brain architecture in autism. *Molecular Psychiatry, 19*(6), 659–667.

Eslami, T., Mirjalili, V., Fong, A., Laird, A. R., & Saeed, F. (2019). ASD-DiagNet: A
hybrid learning approach for detection of autism spectrum disorder using fMRI data.
*Frontiers in Neuroinformatics, 13*, 70.

Fortin, J.-P., Cullen, N., Sheline, Y. I., Taylor, W. D., Aselcioglu, I., Cook, P. A.,
et al. (2018). Harmonization of cortical thickness measurements across scanners and
sites. *NeuroImage, 167*, 104–120.

Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks.
*Advances in Neural Information Processing Systems, 30*, 4878–4887.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern
neural networks. In *Proceedings of the 34th International Conference on Machine
Learning* (pp. 1321–1330).

Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D., Blankertz, B., &
Bießmann, F. (2014). On the interpretation of weight vectors of linear models in
multivariate neuroimaging. *NeuroImage, 87*, 96–110.

Heinsfeld, A. S., Franco, A. R., Craddock, R. C., Buchweitz, A., & Meneguzzi, F.
(2018). Identification of autism spectrum disorder using deep learning and the ABIDE
dataset. *NeuroImage: Clinical, 17*, 16–23.

Holiga, Š., Hipp, J. F., Chatham, C. H., Garces, P., Spooren, W., D'Ardhuy, X. L.,
Bertolino, A., Bouquet, C., Buitelaar, J. K., Bours, C., … Dukart, J. (2019). Patients
with autism spectrum disorders display reproducible functional connectivity
alterations. *Science Translational Medicine, 11*(481), eaat9223.

Hui, S. L., & Walter, S. D. (1980). Estimating the error rates of diagnostic tests.
*Biometrics, 36*(1), 167–171.

Hutchison, R. M., Womelsdorf, T., Allen, E. A., Bandettini, P. A., Calhoun, V. D.,
Corbetta, M., et al. (2013). Dynamic functional connectivity: Promise, issues, and
interpretations. *NeuroImage, 80*, 360–378.

Ilioska, I., Oldehinkel, M., Llera, A., Chopra, S., Looden, T., Chauvin, R., Van
Rooij, D., Floris, D. L., Tillmann, J., Moessnang, C., … Beckmann, C. F. (2023).
Connectome-wide mega-analysis reveals robust patterns of atypical functional
connectivity in autism. *Biological Psychiatry, 94*(1), 29–39.

Jenkinson, M., Bannister, P., Brady, M., & Smith, S. (2002). Improved optimization for
the robust and accurate linear registration and motion correction of brain images.
*NeuroImage, 17*(2), 825–841.

Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in microarray
expression data using empirical Bayes methods. *Biostatistics, 8*(1), 118–127.

Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional
covariance matrices. *Journal of Multivariate Analysis, 88*(2), 365–411.

Leonardi, N., & Van De Ville, D. (2015). On spurious and real fluctuations of dynamic
functional connectivity during rest. *NeuroImage, 104*, 430–436.

Lord, C., Risi, S., Lambrecht, L., Cook, E. H., Leventhal, B. L., DiLavore, P. C., et
al. (2000). The Autism Diagnostic Observation Schedule–Generic: A standard measure of
social and communication deficits associated with the spectrum of autism. *Journal of
Autism and Developmental Disorders, 30*(3), 205–223.

Lord, C., Rutter, M., & Le Couteur, A. (1994). Autism Diagnostic Interview–Revised: A
revised version of a diagnostic interview for caregivers of individuals with possible
pervasive developmental disorders. *Journal of Autism and Developmental Disorders,
24*(5), 659–685.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model
predictions. *Advances in Neural Information Processing Systems, 30*, 4765–4774.

McNemar, Q. (1947). Note on the sampling error of the difference between correlated
proportions or percentages. *Psychometrika, 12*(2), 153–157.

Menon, A., van Rooyen, B., Ong, C. S., & Williamson, R. C. (2015). Learning from
corrupted binary labels via class-probability estimation. In *Proceedings of the 32nd
International Conference on Machine Learning* (pp. 125–134).

Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining well calibrated
probabilities using Bayesian binning. *Proceedings of the AAAI Conference on
Artificial Intelligence, 29*(1), 2901–2907.

Nielsen, J. A., Zielinski, B. A., Fletcher, P. T., Alexander, A. L., Lange, N.,
Bigler, E. D., Lainhart, J. E., & Anderson, J. S. (2013). Multisite functional
connectivity MRI classification of autism: ABIDE results. *Frontiers in Human
Neuroscience, 7*, 599.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et
al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning
Research, 12*, 2825–2830.

Preti, M. G., Bolton, T. A. W., & Van De Ville, D. (2017). The dynamic functional
connectome: State-of-the-art and perspectives. *NeuroImage, 160*, 41–54.

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018).
CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information
Processing Systems, 31*, 6638–6648.

Schulz, M.-A., Bzdok, D., Haufe, S., Haynes, J.-D., & Ritter, K. (2024). Performance
reserves in brain-imaging-based phenotype prediction. *Cell Reports, 43*(1), 113597.

Tzourio-Mazoyer, N., Landeau, B., Papathanassiou, D., Crivello, F., Etard, O.,
Delcroix, N., et al. (2002). Automated anatomical labeling of activations in SPM using
a macroscopic anatomical parcellation of the MNI MRI single-subject brain.
*NeuroImage, 15*(1), 273–289.

Varoquaux, G., Baronnet, F., Kleinschmidt, A., Fillard, P., & Thirion, B. (2010).
Detection of brain functional-connectivity difference in post-stroke patients using
group-level covariance modeling. In *Medical Image Computing and Computer-Assisted
Intervention—MICCAI 2010* (pp. 200–208).

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical
inference. *Journal of the American Statistical Association, 22*(158), 209–212.

Yeo, B. T. T., Krienen, F. M., Sepulcre, J., Sabuncu, M. R., Lashkari, D.,
Hollinshead, M., et al. (2011). The organization of the human cerebral cortex
estimated by intrinsic functional connectivity. *Journal of Neurophysiology, 106*(3),
1125–1165.
