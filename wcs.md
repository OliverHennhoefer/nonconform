# Model-free selective inference under covariate shift via weighted conformal p-values

**Ying Jin**$^{1}$, **Emmanuel J. Candès**$^{1,2}$

$^{1}$Department of Statistics, Stanford University
$^{2}$Department of Mathematics, Stanford University

## Abstract

This paper introduces novel weighted conformal p-values and methods for model-free selective inference. The problem is as follows: given test units with covariates $X$ and missing responses $Y$, how do we select units for which the responses $Y$ are larger than user-specified values while controlling the proportion of false positives? Can we achieve this without any modeling assumptions on the data and without any restriction on the model for predicting the responses? Last, methods should be applicable when there is a covariate shift between training and test data, which commonly occurs in practice.

We answer these questions by first leveraging any prediction model to produce a class of well-calibrated weighted conformal p-values, which control the type-I error in detecting a large response. These p-values cannot be passed onto classical multiple testing procedures since they may not obey a well-known positive dependence property. Hence, we introduce weighted conformalized selection (WCS), a new procedure which controls false discovery rate (FDR) in finite samples. Besides prediction-assisted candidate selection, WCS (1) allows to infer multiple individual treatment effects, and (2) extends to outlier detection with inlier distributions shifts. We demonstrate performance via simulations and applications to causal inference, drug discovery, and outlier detection datasets.

## 1 Introduction

Many scientific discovery and decision making tasks concern the selection of promising candidates from a potentially large pool. In drug discovery, scientists aim to find drug candidates—from a huge chemical space of molecules or compounds—with sufficiently high binding affinity to a target. College admission officers or hiring managers seek applicants with a high potential of excellence. In healthcare, one would like to allocate resources to individuals who would benefit the most. The list of such examples is endless. In all these problems, a decision maker would like to identify those units for which an unknown outcome of interest (affinity for a target molecule/job performance/reduction in stress level) takes on a large value.

Machine learning has shown tremendous potential for accelerating these resource- and time-consuming processes whereby predictions for the unknown outcomes are used to shortlist promising candidates before any further in-depth investigation \cite{lavecchia2013virtual,carpenter2018machine,sajjadiani2019using}. This is a sharp departure from traditional strategies that either directly evaluate the outcomes via a physical screening to determine drug binding affinities, or “predict” the unknown outcomes via human judgement in healthcare or job hiring applications. In the new paradigm, a prediction model draws evidence from training data to inform values the unknown outcome may take on given the features of a unit.

If the intent is thus to use machine learning to screen candidates, we would need to understand and control the uncertainty in black-box prediction models as a selection tool. First, gathering evidence for new discoveries from past instances is challenging as they are not necessarily similar; i.e. they may be drawn from distinct distributions. Second, since a subsequent investigation applied to the shortlisted candidates is often resource intensive, it is meaningful to limit the *error rate on the selected*. It is not so much the accuracy of the prediction that matters, rather the proportion of selected units not passing the test. Keeping such a proportion at sufficiently low levels is a non-traditional criterion in typical prediction problems.

This paper studies the reliable selection of promising units/individuals whose unknown outcomes are sufficiently large. We develop methods that apply in a model-free fashion—without making any modeling assumptions on the data generating process while controlling the expected proportion of false discoveries among the selected.

### 1.1 Reliable selection under distribution shift

Formally, suppose test samples $\{(X_{j},Y_{j})\}_{j\in\mathcal{D}_{\textnormal{test}}}$ are drawn i.i.d. from an unknown distribution $Q$, whose outcomes $\{Y_j\}_{j\in\mathcal{D}_{\textnormal{test}}}$ are unobserved. Our goal is to identify a subset $\mathcal{R}\subseteq \mathcal{D}_{\textnormal{test}}$ with outcomes above some known, potentially random, thresholds $\{c_{ j}\}_{j\in \mathcal{D}_{\textnormal{test}}}$. To draw statistical evidence for large outcomes in $\mathcal{D}_{\textnormal{test}}$, we assume we hold an independent set of i.i.d. calibration data $\{(X_i,Y_i)\}_{i\in \mathcal{D}_{\textnormal{calib}}}$ whose outcomes are, this time, observed.

Before we begin our discussion, we note that a challenge that underlies almost all applications is that samples in $\mathcal{D}_{\textnormal{calib}}$ are often different from those in $\mathcal{D}_{\textnormal{test}}$. That is, in many cases, a point $(X_{j},Y_{j})$ in $\mathcal{D}_{\textnormal{test}}$ is sampled from $Q$ while a point $(X_{i},Y_{i})$ in $\mathcal{D}_{\textnormal{calib}}$ is sampled from a different distribution $P$. We elaborate on the distribution shift issue by considering two motivating examples.

**Drug discovery.** Virtual screening, which identifies viable drug candidates using predicted affinity scores from supervised machine learning models, is increasingly popular in early stages of drug discovery \cite{huang2007drug,koutsoukas2017deep,vamathevan2019applications,dara2021machine}. In this application, $\mathcal{D}_{\textnormal{calib}}$ is the set of drugs with known affinities, while $\mathcal{D}_{\textnormal{test}}$ is the set of new drugs whose binding affinities are of interest. In practice, there usually is a distribution shift between the two batches: for instance, an experimenter may favor a specific structure when choosing candidates to physically screen their true binding affinities \cite{polak2015early}. Since physically screened drugs are subsequently added to the calibration dataset $\mathcal{D}_{\textnormal{calib}}$, they may not be representative of the new drugs in $\mathcal{D}_{\textnormal{test}}$. Accounting for such distribution shifts when applying virtual screening is crucial for the reliability of the whole drug discovery process, and has attracted recent attention \cite{Krstajic2021}.

**College admission.** A college may be interested in prospective students who are likely to graduate after four years of undergraduate education (a binary outcome of interest) among all applicants in $\mathcal{D}_{\textnormal{test}}$ as to maintain a reasonable graduation rate. The issue is that $\mathcal{D}_{\textnormal{calib}}$ usually consists of students who *were* admitted in the past as the outcome is only known for these students. Since previous cohorts may have been selected by applying various admission criteria, the distribution of the training data is different from that of the new applicants even if the distribution of applicants does not much vary over the years. Similar concerns apply to job hiring when using prediction models to select suitable applicants \cite{faliagka2012application,shehu2016adaptive}; the way an institution chooses to record the outcomes of former candidates can lead to a discrepancy between the documented candidates and those under consideration.

In this paper, we propose to address the distribution shift issue by considering a scenario where the shift can be entirely attributed to observed covariates; this is commonly referred to as a covariate shift in the literature \cite{sugiyama2007covariate,tibshirani2019conformal}. Concretely, we assume that there is some function $w\colon \mathcal{X}\to \mathbb{R}^+$ such that
\begin{align}
\label{eq:cov_shift}
\frac{\mathrm{d} Q}{\mathrm{d} P}(x,y) = w(x),\quad  P\textrm{-almost surely},
\end{align}
where $ \mathrm{d} Q/\mathrm{d} P $ denotes the Radon-Nikodym derivative. While all kinds of distribution shift could happen in practice, covariate shift—widely used to characterize distribution shifts caused by selection on known features—covers many important cases. In drug discovery, if the preference for choosing which samples to screen only depends upon the covariates, e.g. the sequence of amino acids of the compound or the physical properties of the compound, we are dealing with the covariate shift \eqref{eq:cov_shift}. In college admission or job hiring, if a specific type of applicant—characterized by certain features—were favored in previous admission/hiring cycles, such a preference would yield the shift \eqref{eq:cov_shift} between previous students/employees and current applicants.

Returning to our decision problem, we are interested in finding as many units $j\in \mathcal{D}_{\textnormal{test}}$ as possible with $Y_{j}>c_j$, while controlling the false discovery rate (FDR) below some pre-specified level $q\in (0,1)$. The FDR is the expected fraction of false discoveries, where a false discovery is a selected unit for which $Y_j \le c_j$, i.e., the outcome turns out to fall below the threshold. Formally, we wish to find $\mathcal{R}$ obeying
\begin{align}
\label{eq:fdr}
\textnormal{FDR} := \mathbb{E}\Bigg[  \frac{\sum_{j\in\mathcal{D}_{\textnormal{test}}} \mathbb{1}\{j\in \mathcal{R},Y_{j}\leq c_{ j}\}}{\max\{1,|\mathcal{R}|\}}  \Bigg] \leq q, 
\end{align}
where $q$ is the nominal target FDR level. The expectation in \eqref{eq:fdr} is taken over both the new test samples and the calibration data/screening procedure. The FDR quantifies the tradeoff between resources dedicated to the shortlisted candidates and the benefits from true positives \cite{benjamini1995controlling}. By controlling the FDR, we ensure that a sufficiently large proportion of follow-up resources—such as clinical trials in drug discovery, human evaluation of student profiles and interviews—to confirm predictions from machine learning models are devoted to interesting candidates.

The problem of FDR-controlled selection with machine learning predictions has been studied in an earlier paper \cite{jin2022selection} under the assumption of no distribution shift. This means the new instances are exchangeable with the known calibration data, which, as we discussed above, is rarely the case in practice. Therefore, there is a pressing need for novel techniques to address the distribution shift issue.

### 1.2 Preview of theoretical contributions

Our strategy consists in (1) constructing a calibrated confidence measure (p-value), which applies to any predictive model, for detecting a large unknown outcome, and (2) in employing multiple testing ideas to build the selection set from test p-values.

**Weighted conformal p-value: a calibrated confidence measure.**
We introduce random variables $\{p_j\}_{j\in \mathcal{D}_{\textnormal{test}}}$, which resemble p-values in the sense that for each $j\in \mathcal{D}_{\textnormal{test}}$, they obey
\begin{align}
\label{eq:general_pvalue}
\mathbb{P}(p_j \leq t, ~ Y_{j} \leq c_{j} ) \leq t,\quad \textrm{for all}~t\in [0,1],  
\end{align}
where the probability is over both the calibration data and the test sample $(X_{j},Y_{j},c_{j})$. This is similar to the definition of a classical p-value, hence, we call $p_j$ a “p-value”. With this, for any $\alpha\in(0,1)$, selecting $j$ if and only if $p_j\leq \alpha$ controls the type-I error below $\alpha$ (the chance that $Y_j$ is below threshold is at most $\alpha$). Note however that \eqref{eq:general_pvalue} is perhaps an unconventional notion of type-I error, since it accounts for the randomness in the “null hypothesis” $H_{j}\colon Y_{j}\leq c_{j}$.

We construct $p_j$ using ideas from conformal prediction \cite{vovk2005algorithmic,tibshirani2019conformal}. Take any monotone function $V\colon \mathcal{X}\times\mathcal{Y}\to \mathbb{R}$ (see Definition \ref{def:monotone}) built upon any prediction model that is independent of $\mathcal{D}_{\textnormal{calib}}$ and $\mathcal{D}_{\textnormal{test}}$. A concrete instance is $V(x,y) = y - \widehat{\mu}(x)$, where $\widehat{\mu}\colon \mathcal{X}\to \mathcal{Y}$ is any predictor trained on a held-out dataset. With scores $V_i = V(X_i,Y_i)$, $i\in \mathcal{D}_{\textnormal{calib}}$, and weight function $w(\cdot)$ given by \eqref{eq:cov_shift}, we construct $p_j$ to be approximately equal to $\widehat{F}(\widehat{V}_j)$, where $\widehat{F}(\cdot)$ is the cumulative distribution function (c.d.f.) of the empirical distribution
$$
\frac{ \sum_{i \in \mathcal{D}_{\textnormal{calib}}} \,  w(X_i) \, \delta_{V_i} }{  \sum_{i \in \mathcal{D}_{\textnormal{calib}}} w(X_i) }.
$$
While the precise definition of $p_j$ is in Section \ref{sec:pval}, Figure \ref{fig:pvalue} visualizes the idea.

Intuitively, $p_j$ contrasts the value of $\widehat{V}_{j}=V(X_{j},c_{j})$ against the distribution of the unknown `oracle' score $V_j=V(X_j,Y_j)$. The latter is approximated by $\widehat{F}(\cdot)$, which reweights the training data with weights reflecting the covariate shift \eqref{eq:cov_shift}. As such, $\widehat{F}(V_j)$ is approximately uniformly distributed in $[0,1]$. Thus, a notably small value of $p_j\approx \widehat{F}(\widehat{V}_j)$ relative to the uniform distribution informs that $\widehat{V}_j$ is smaller than what is typical of $V_j$, which further suggests evidence for $Y_j\geq c_j$. Leveraging conformal prediction techniques, such high-level ideas are made exact to achieve \eqref{eq:general_pvalue}.

**Figure 1 Caption:** Visualization of weighted conformal p-values. Left: Distribution of scores in the calibration set (gray) and individual values of $w(X_i)$ (orange) for $i\in\mathcal{D}_{\textnormal{calib}}$. Right: Weighted empirical c.d.f. of $\{V_i\}_{i\in \mathcal{D}_{\textnormal{calib}}}$; the red dashed lines illustrate how $p_j$ is computed from $\widehat{V}_j$, in which $j\in \mathcal{D}_{\textnormal{test}}$.
\label{fig:pvalue}

**Intricate dependence among weighted conformal p-values.**

Going beyond per-unit type-I error control, selecting from multiple test samples requires dealing with multiple p-values $\{p_j\}_{j\in \mathcal{D}_{\textnormal{test}}}$. A natural idea is then to use multiple testing ideas as if $\{p_j\}$ were classical p-values. However, the situation is complicated here because $\{p_j\}_{j\in \mathcal{D}_{\textnormal{test}}}$ all depend on the same set of calibration data, and, in sharp contrast to classical multiple testing, there is another source of randomness in the unknown outcomes.

Our second theoretical result shows the mutual dependence among multiple p-values is particularly challenging. It states that the favorable positive dependence structure among p-values, which was the key to FDR control with other p-values derived with conformal prediction ideas \cite{bates2021testing,jin2022selection}, can be violated in the presence of data-dependent weights in \eqref{eq:def_wcpval_rand}. As a result, it remains unclear whether applying existing multiple testing procedures controls the FDR in finite samples.

**Weighted Conformalized Selection (WCS).**
Last, we introduce WCS, a new procedure that achieves finite-sample FDR control under distribution shift. The idea is to calibrate the selection threshold for each $j\in \mathcal{D}_{\textnormal{test}}$ using a set of “auxiliary” p-values $\{\bar{p}_\ell^{(j)}\}_{\ell\in \mathcal{D}_{\textnormal{test}}}$. These p-values happen to obey a special conditional independence property, namely,
\begin{align*}
\bar p_j^{(j)}  \perp\!\!\!\perp \{\bar p_\ell^{(j)}\}_{\ell\neq j} \mid \mathcal{Z}_j,
\end{align*}
where $\mathcal{Z}_j$ is the unordered set of the calibration data and the $j$-th test sample. We show that FDR control is valid regardless of the machine learning model used in constructing $V(\cdot,\cdot)$, and applies to a wide range of scenarios where the cutoffs $c_{j}$ can themselves be random variables.

### 1.3 Broader scope

Our methods happen to be applicable to additional statistical inference problems.

**Multiple counterfactual inference.**
The individual treatment effect (ITE) is a random variable that characterizes the difference between an individual’s outcome when receiving a treatment versus not, whose inference relies on comparing the observed outcome with the counterfactual \cite{imbens2015causal,lei2021conformal,jin2023sensitivity}. To infer the counterfactual of a unit under treatment, the calibration data are units who do not receive the treatment. If the treatment assignment depends on the covariates, then there may be, and usually will be, a covariate shift between treated and control units. We discuss this problem in Section \ref{sec:ite}.

**Outlier detection.** Imagine we have access to i.i.d. inliers $\{Z_i\}_{i\in \mathcal{D}_{\textnormal{calib}}}$ from an unknown distribution $P$. Then outlier detection \cite{bates2021testing} seeks to find outliers among new samples $\{Z_j\}_{j\in \mathcal{D}_{\textnormal{test}}}$, i.e., those which do not follow the inlier distribution $P$. A potential application is fraud detection in financial transactions, where controlling the FDR ensures reasonable resource allocation in follow-up inquiries. However, financial behavior varies with users and it is possible that normal transactions of interest follow different distributions across different populations. A related problem is to detect concept drifts; that is, allowing a subset of features $X\subseteq Z$ to follow a distinct distribution, and only detecting test samples whose $Z\mid X$ distribution differs from $P_{Z\mid X}$. We discuss this problem in Section \ref{sec:outlier}.

As in \eqref{eq:general_pvalue}, the FDR \eqref{eq:fdr} is marginalized over the randomness in the hypotheses. To this end, WCS requires all $\{X_{n+j},Y_{n+j}\}_{j=1}^m$ to be i.i.d. draws from a super-population. To relax this, we generalize our methods to control a notion of FDR that is conditional on the random hypotheses. This is useful for handling imbalanced data in classification, and the aforementioned outlier detection problem.

### 1.4 Data and code

An R package, `ConfSelect`, that implements Weighted Conformalized Selection (as well as the version without weights) together with code to reproduce all experiments in this paper, can be found in the GitHub repository [https://github.com/ying531/conformal-selection](https://github.com/ying531/conformal-selection).

### 1.5 Related work

Our methods build upon the conformal inference framework \cite{vovk2005algorithmic} to infer unknown outcomes. There, the theoretical guarantee for conformal prediction sets is usually marginally valid over the randomness in a single test point. As discussed in \cite{jin2022selection}, one might be interested in multiple test samples simultaneously; in such situations, standard conformal prediction tools are insufficient.

This work is connected to a recent line of research on selective inference issues arising in predictive inference. Some works use conformal p-values (whose definition differs from ours) with `full' observations $\{(X_{j},Y_j)\}_{j\in \mathcal{D}_{\textnormal{test}}}$—that is, the response is observed—for outlier detection, i.e., for identifying whether $(X_{j},Y_j)$ follows the same distribution as the calibration data $\{(X_{i},Y_i)\}_{i\in \mathcal{D}_{\textnormal{calib}}}$ \cite{bates2021testing,liang2022integrative,marandon2022machine}. Whereas these methods do not consider distribution shift between calibration and test inliers, Section \ref{sec:outlier} extends all this. Here, the response $Y_{n+j}$—the object of inference—is not observed and, therefore, the problem is completely different. Consequently, methods and techniques to deal with this are new.

Our focus on FDR control is similar in spirit to other recent papers; for instance, rather than marginal coverage of prediction sets, \cite{fisch2022conformal} studies the number of false positives, and \cite{bao2023selective} the miscoverage rate on selected units.

Our methodology draws ideas from the multiple hypothesis testing literature, where the FDR is a popular type-I error criterion in exploratory analysis for controlling the proportion of `false leads' in follow-up studies \cite{benjamini1995controlling,benjamini2001control}. This paper is distinct from the conventional setting. First, testing a random outcome instead of a model parameter leads to a random null set and a complicated dependence structure. Second, our inference relies on (weighted) exchangeability of the data while imposing no assumptions on their distributions. In contrast, a null hypothesis often specifies the distribution of test statistics or p-values. Later on, we shall also draw connections to a recent literature dealing with dependence in multiple testing, such as that concerned with e-values \cite{vovk2021values,wang2020false} and conditional calibration of the BH procedure \cite{fithian2020conditional}.

The application of our method is related to a substantial literature that emphasizes the importance of uncertainty quantification in drug discovery \cite{Norinder2014,svensson2017improving,Ahlberg2017current,svensson2018conformal,cortes2019concepts,wang2022improving}. Although existing works aim to employ conformal inference to assign confidence to machine prediction, this is often done in heuristic ways, which potentially invalidates the guarantees of conformal prediction due to selection issues. Instead, we provide here a principled approach to prioritizing drug candidates with clear and interpretable error control (i.e., guaranteeing a sufficiently high `hit rate' \cite{wang2022improving}), and offer a potential solution to the widely recognized distribution shift problem \cite{Krstajic2021,fannjiang2023novelty}.

The application to individual treatment effects in Section \ref{sec:ite} is connected to \cite{caughey2021randomization}; the distinction is that they focus on a summary statistic such as a population quantile of the ITEs, while our method detects individuals with positive ITEs. Also related is \cite{duan2021interactive}, which tests individuals for positive treatment effects with FDR control; however, in their setting a positive ITE means the whole distribution of the control outcome is dominated by that of the treated outcome, whereas we directly compare two random variables. Accordingly, our techniques are very different from these two references.

## 2 Weighted conformal p-values
\label{sec:pval}

### 2.1 Construction of weighted conformal p-values

Our weighted conformal p-values build upon any *monotone* score function, defined as follows.

**Definition 2.1 (Monotonicity).**
A score function $V(\cdot,\cdot)\colon \mathcal{X}\times\mathcal{Y}\to \mathbb{R}$ is monotone if $V(x,y)\leq V(x,y')$ holds for any $x\in \mathcal{X}$ and any $y,y'\in \mathcal{Y}$ obeying $y\leq y'$.
\label{def:monotone}

Intuitively, the score function (often referred to as nonconformity score in conformal prediction \cite{vovk2005algorithmic}) describes how well a hypothesized value $y\in \mathcal{Y}$ *conforms* to the machine prediction. A popular and monotone nonconformity score is $V(x,y) = y-\widehat{\mu}(x)$, where $\widehat{\mu}\colon\mathcal{X}\to \mathbb{R}$ is a point prediction; other choices include those based on quantile regression \cite{romano2019conformalized} or estimates of the conditional c.d.f. \cite{chernozhukov2021distributional}.

From now on, write $\mathcal{D}_{\textnormal{calib}} = \{1,\dots,n\}$ and $\mathcal{D}_{\textnormal{test}}=\{n+1,\dots,n+m\}$. Compute $V_i=V(X_i,Y_i)$ for $i=1,\dots,n$ and $\widehat{V}_{n+j}=V(X_{n+j},c_{n+j})$ for $j=1,\dots,m$. Our weighted conformal p-values are defined as
\begin{align}
\label{eq:def_wcpval_rand}
p_j = \frac{\sum_{i=1}^n w(X_i)\mathbb{1} {\{V_i <\widehat{V}_{n+j} \}}+  ( w(X_{n+j}) + \sum_{i=1}^n w(X_i)\mathbb{1} {\{V_i = \widehat{V}_{n+j} \}})\cdot U_j}{\sum_{i=1}^n w(X_i) + w(X_{n+j})},
\end{align}
where $w(\cdot)$ is the covariate shift function in \eqref{eq:cov_shift}, and $U_j\stackrel{\textnormal{i.i.d.}}{\sim} \textrm{Unif}([0,1])$ are tie-breaking random variables. When $w(\cdot)\equiv 1$, our p-values reduce to the conformal p-values in \cite{jin2022selection}, see Appendix \ref{app:recap_unweighted}. We also refer readers to Appendix \ref{app:connection_conformal} for the connection between our p-values and conformal prediction intervals.

Roughly speaking, with a monotone score, the p-value \eqref{eq:def_wcpval_rand} measures how small $c_{n+j}$ is compared with typical values of $Y_{n+j}$, and provides calibrated evidence for a large outcome as expressed in \eqref{eq:general_pvalue}. This can be derived using conformal inference theory \cite{tibshirani2019conformal} and the monotocity of $V$. We include a formal result here for completeness, whose proof is in Appendix \ref{app:lem_general_pval}.

**Lemma 2.2.**
The tail inequality \eqref{eq:general_pvalue} holds if the covariate shift \eqref{eq:cov_shift} holds and the score function is monotone.
\label{lem:general_pval}

Consequently, selecting a unit if $p_j \le \alpha$ controls the type-I error at level $\alpha$ for each $j\in \mathcal{D}_{\textnormal{test}}$. We shall however see that dealing with multiple test samples is far more challenging.

### 2.2 Weighted conformal p-values are not PRDS
\label{subsec:no_PRDS}

Applying multiple testing procedures to our p-values to obtain a selection set naturally comes to mind. Previous works applying the Benjamini-Hochberg (BH) procedure \cite{benjamini1995controlling} to p-values obtained via conformal inference ideas indeed control the FDR \cite{bates2021testing,jin2022selection,marandon2022machine}. This is because the p-values obey a favorable dependence structure called *positive dependence on a subset* (PRDS).

**Definition 2.3 (PRDS).**
A random vector $X=(X_1,\dots,X_m)$ is PRDS on a subset $\mathcal{I}$ if for any $i\in \mathcal{I}$ and any increasing set $D$, the probability $\mathbb{P}(X\in D\mid X_i=x)$ is increasing in $x$.
\label{def:prds}

Above, a set $D\subseteq \mathbb{R}^m$ is *increasing* if $a\in D$ and $b\succeq a$ implies $b\in D$ ($b \succeq a$ means that all the components of $b$ are larger than or equal to those of $a$). It is well-known that the BH procedure controls the FDR when applied to PRDS p-values \cite{benjamini2001control}.
The PRDS property among (unweighted) conformal p-values was first studied in \cite{bates2021testing}, and generalized in \cite{jin2022selection} to plug-in values $c_{n+j}$ in lieu of $Y_{n+j}$, thereby forming the basis for FDR control.
If the PRDS property were to hold for the weighted conformal p-values, then applying the BH procedure would also control the FDR. This is however not always the case; a constructive proof of Proposition \ref{prop:counter_PRDS} is in Appendix \ref{app:subsec_prds}.

**Proposition 2.4.**
There exist a sample size $n$, a weight function $w\colon \mathcal{X}\to \mathbb{R}^+$, a nonconformity score $V\colon \mathcal{X}\times\mathcal{Y}\to \mathbb{R}$, and distributions $P$ and $Q$ obeying \eqref{eq:cov_shift}, such that the weighted conformal p-values \eqref{eq:def_wcpval_rand} are not PRDS for $\{X_i,Y_i\}_{i=1}^n\stackrel{\textnormal{i.i.d.}}{\sim} P$ and $\{X_{n+j},Y_{n+j}\}_{j=1}^m\stackrel{\textnormal{i.i.d.}}{\sim} Q$.
\label{prop:counter_PRDS}

A little more concretely, we find that the PRDS property is likely to fail when the nonconformity scores are negatively correlated with the weights.
Note that the dependence among conformal p-values arises from sharing the calibration data. Under exchangeability (i.e. taking equal weightes $w(x)\equiv 1$), a smaller conformal p-value is (only) associated with larger calibration scores, and hence smaller p-values for other test points. However, with negatively associated weights and scores, a smaller p-value may also be due to smaller calibration weights—instead of larger calibration scores—which implies that other p-values may take on larger values.

### 2.3 BH procedure controls FDR asymptotically

Before introducing our new selection methods, we nevertheless show *asymptotic* FDR control using the BH procedure.

**Theorem 2.5.**
Suppose $w(\cdot)$ is uniformly bounded by a fixed constant, the covariate shift \eqref{eq:cov_shift} holds, and $\{c_{n+j}\}_{j=1}^m$ are i.i.d. random variables. Fix any $q\in(0,1)$, and let $\mathcal{R}$ be the output of the BH procedure applied to $\{p_j\}_{j=1}^m$ in \eqref{eq:def_wcpval_rand}. Then the following results hold.
1. For any fixed $m$, it holds that $\limsup_{n\to \infty} \textnormal{FDR} \leq q$.
2. Suppose $m,n\to \infty$. Under a mild technical condition (see Appendix \ref{app:fdr_asymp}), $\limsup_{m,n\to\infty} \textnormal{FDR} \leq q$. Furthermore, the asymptotic FDR and power of BH can be exactly characterized in this case.
\label{thm:fdr_asymp}

We provide the complete technical version of Theorem \ref{thm:fdr_asymp} in Theorem \ref{thm:fdr_asymp_full}, whose proof is in Appendix \ref{app:thm_fdr_asymp}.
In fact, we prove that as $n\to \infty$, the weighted conformal p-values converge to i.i.d. random variables that are dominated by the uniform distribution, which ensures asymptotic FDR control (see Appendix \ref{app:thm_fdr_asymp} for details). The technical condition we impose resembles that in \cite{storey2004strong}, which ensures the existence of a limit point of the rejection threshold and enables the asymptotic analysis.

We also remark that the PRDS property is a sufficient, but not necessary, condition for the BH procedure to control the FDR. In fact, we will see that the BH procedure with weighted conformal p-values empirically controls the FDR in all of our numerical experiments, hence it remains a reasonable option in practice.

## 3 Finite-sample FDR control
\label{sec:method}

We now introduce a new multiple testing procedure, WCS, that controls the FDR in finite samples with weighted conformal p-values. Our method builds on the following p-values:
\begin{align}
\label{eq:weighted_pval}
{p}_j = \frac{\sum_{i=1}^n w(X_i)\mathbb{1}{\{V_i < \widehat{V}_{n+j} \}}+  w(X_{n+j}) }{\sum_{i=1}^n w(X_i) + w(X_{n+j})}.
\end{align}
It is a slight modification of \eqref{eq:def_wcpval_rand} up to random tie-breaking. The asymptotic results in Theorem \ref{thm:fdr_asymp} also hold with these p-values as long as the distributions of the $V_i$'s and $\widehat{V}_{n+j}$'s do not have point masses.

### 3.1 Weighted Conformalized Selection

As before, we compute $V_i=V(X_i,Y_i)$ for $i=1,\dots,n$ and $\widehat{V}_{n+j} = V(X_{n+j},c_{n+j})$ for $j=1,\dots,m$. For each $j$, we compute a set of auxiliary p-values
\begin{align}
\label{eq:mod_pval}
{p}_{\ell}^{(j)} := \frac{\sum_{i=1}^n w(X_i) \mathbb{1} {\{V_i < \widehat{V}_{n+\ell}\}} + w(X_{n+j}) \mathbb{1} {\{ \widehat{V}_{n+j}<  \widehat{V}_{n+\ell}\}} }{\sum_{i=1}^n w(X_i) + w(X_{n+j})}, \quad \ell\neq j.
\end{align}
Then, we let $\widehat{\mathcal{R}}_{j\to 0}$ be the rejection set of BH applied to $\{  p_1^{(j)},\dots,  p_{j-1}^{(j)}, 0,   p_{j+1}^{(j)},\dots,  p_{m}^{(j)}\}$ at the nominal level $q$. Note that $j\in \widehat{\mathcal{R}}_{j\to 0}$ by default, and set $s_j = \frac{q|\widehat{\mathcal{R}}_{j\to 0}  | }{m}$. We then compute the first-step rejection set $\mathcal{R}^{(1)}:=\{j \colon   p_j \leq s_j\}$. Finally, we prune $\mathcal{R}^{(1)}$ to obtain the final selection set $\mathcal{R}$ using any of the following three methods.

(a) *Heterogeneous pruning*: generate i.i.d. random variables $\xi_j \sim \textrm{Unif}[0,1]$, and set
\begin{align}
\label{eq:R_hete}
\mathcal{R} := \mathcal{R}_{\textnormal{hete}} = \Big\{ j\colon  p_j \leq s_j, ~ \xi_j |\widehat{\mathcal{R}}_{j\to 0} | \leq  r_{\textnormal{hete}}^*  \Big\},
\end{align}
where
$
r_{\textnormal{hete}}^* := \max\big\{ r\colon  \sum_{j=1}^m \mathbb{1} {\{  p_j\leq s_j, \, \xi_j   |\widehat{\mathcal{R}}_{j\to 0}  | \leq r \}}   \geq r \big\}.
$

(b) *Homogeneous pruning*: generate an independent $\xi\sim \textrm{Unif}[0,1]$, and set
\begin{align}
\label{eq:R_homo}
\mathcal{R} :=  \mathcal{R}_{\textnormal{homo}} = \Big\{ j\colon  p_j \leq s_j, ~ \xi  |\widehat{\mathcal{R}}_{j\to 0} | \leq  r_{\textnormal{homo}}^*  \Big\},
\end{align}
where
$
r_{\textnormal{homo}}^* := \max\big\{ r\colon  \sum_{j=1}^m \mathbb{1} {\{ p_j\leq s_j, \, \xi   |\widehat{\mathcal{R}}_{j\to 0}  | \leq r \}}   \geq r \big\}.
$

(c) *Deterministic pruning*: define the rejection set
\begin{align}
\label{eq:R_dete}
\mathcal{R} := \mathcal{R}_{\textnormal{dtm}} = \Big\{ j\colon p_j \leq s_j, ~  |\widehat{\mathcal{R}}_{j\to 0} | \leq  r_{\textnormal{dtm}}^*  \Big\},
\end{align}
where
$
r_{\textnormal{dtm}}^* := \max\big\{ r\colon  \sum_{j=1}^m \mathbb{1} {\{ p_j\leq s_j, \, |\widehat{\mathcal{R}}_{j\to 0}  | \leq r \}}   \geq r \big\}.
$
In all options, we use $p_j$ defined in \eqref{eq:weighted_pval}.

It is straightforward to see that $\mathcal{R}_{\textnormal{dtm}}\subseteq \mathcal{R}_{\textnormal{hete}}$ and $\mathcal{R}_{\textnormal{dtm}}\subseteq \mathcal{R}_{\textnormal{homo}}$, i.e., random pruning leads to larger rejection sets. The selection procedure is summarized in Algorithm \ref{alg:bh}.

**Algorithm 1: Weighted Conformalized Selection**
\label{alg:bh}
**Input:** Calibration data $\{(X_i,Y_i)\}_{i=1}^n$, test data  $\{X_{n+j}\}_{j=1}^m$, thresholds $\{c_{n+j}\}_{j=1}^m$, weight $w(\cdot)$, FDR target $q\in(0,1)$, monotone nonconformity score $V\colon \mathcal{X}\times\mathcal{Y}\to \mathbb{R}$, pruning method $\in\{\texttt{hete}, \texttt{homo}, \texttt{dtm}\}$.

1. Compute $V_i = V(X_i,Y_i)$ for $i=1,\dots,n$, and $\widehat{V}_{n+j}= V(X_{n+j},c_{n+j})$ for $j=1,\dots,m$.
2. Construct weighted conformal p-values $\{ p_j\}_{j=1}^m$ as in \eqref{eq:weighted_pval}.

*-- First-step selection --*
3. **for** $j=1,\dots,m$ **do**
4. Compute p-values $\{ {p}_\ell^{(j)}\}$ as in \eqref{eq:mod_pval}.
5. (BH procedure) Compute $k^*_j = \max\big\{k\colon 1 +\sum_{\ell\neq j} \mathbb{1}\{{p}_\ell^{(j)}\leq qk/m\}\geq k\big\}$.
6. Compute $\widehat{\mathcal{R}}_{j\to 0} = \{j\}\cup\{\ell \neq j\colon  {p}_\ell^{(j)}\leq q k^*_j /m\}$.
7. **end for**
8. Compute the first-step selection set $\mathcal{R}^{(1)} = \{j\colon  {p}_j \leq q|\widehat{\mathcal{R}}_{j\to 0}|/m\}$.

*-- Second-step pruning --*
9. Compute $\mathcal{R} = \mathcal{R}_{\textrm{hete}}$ as in \eqref{eq:R_hete} or $\mathcal{R} = \mathcal{R}_{\textrm{homo}}$ as in \eqref{eq:R_homo} or $\mathcal{R} = \mathcal{R}_{\textrm{dtm}}$ as in \eqref{eq:R_dete}.

**Output:** Selection set $\mathcal{R}$.

### 3.2 Theoretical guarantee

The following theorem, whose proof is in Appendix \ref{app:thm_calib_ite}, shows exact finite-sample FDR control.

**Theorem 3.1.**
Write $Z_i=(X_i,Y_i)$ for $i=1,\dots,m+n$, and $\widetilde{Z}_{n+j}=(X_{n+j},c_{n+j})$ for $j=1,\dots,m$. Suppose $\{Z_i\}_{i=1}^n\stackrel{\textnormal{i.i.d.}}{\sim} P$ and $\{Z_{n+j}\}_{j=1}^m\stackrel{\textnormal{i.i.d.}}{\sim} Q$, and \eqref{eq:cov_shift} holds for the input weight function $w(\cdot)$ of Algorithm \ref{alg:bh}. Assume that for each $j=1,\dots,m$, the samples in $\{Z_1,\dots,Z_n,Z_{n+j}\}\cup\{\widetilde{Z}_{n+\ell}\}_{\ell\neq j}$ are mutually independent. Then with either $\mathcal{R} \in \{\mathcal{R}_{\textnormal{hete}}, \mathcal{R}_{\textnormal{homo}}, \mathcal{R}_{\textnormal{dtm}}\}$, it holds that
\begin{align*}
\mathbb{E}\Bigg[  \frac{\sum_{j=1}^m \mathbb{1} {\{j \in \mathcal{R}, Y_{n+j}\leq c_{n+j}\}} }{1\vee |\mathcal{R}|} \Bigg]   \leq q 
\end{align*}
(the expectation is taken over both calibration and test data).
\label{thm:calib_ite}

As we allow for random thresholds $c_{n+j}$, the independence assumption rules out the cases where the thresholds $c_{n+j}$ are adversarially chosen; the reason is that they would break certain exchangeability properties between scores $\{\widehat{V}_{n+\ell}\}_{\ell=1}^m$ and calibration scores $\{V_{i}\}_{i=1}^n$. Independence holds if $c_{n+j}$ is pre-determined, or more generally, if $c_{n+j}$ is a random variable associated with independent test samples (such as when $\{X_{n+j},c_{n+j},Y_{n+j}\}_{j=1}^m$ are i.i.d. tuples that are independent of the calibration data). Examples include individual treatment effects studied in Section \ref{sec:ite}, and the drug-target interaction prediction task studied in Section \ref{subsec:dti}. Other examples with random thresholds related to healthcare are in \cite[Section 2.4]{jin2022selection}.

While the $p_j$'s are still marginally stochastically larger than uniforms, their complicated dependence requires additional care, and this is why we use the auxiliary p-values $\{p_\ell^{(j)}\}$ as `calibrators' to determine a potentially different rejection rule than naively applying the BH procedure. The proof of Theorem \ref{thm:calib_ite} relies on comparing our p-values \eqref{eq:weighted_pval} to oracle p-values
\begin{align}
\label{eq:orc_w_pval_nr}
\bar p_j  = \frac{\sum_{i=1}^n w(X_i)\mathbb{1} {\{V_i < {V}_{n+j} \}}+  w(X_{n+j})  }{\sum_{i=1}^n w(X_i) + w(X_{n+j})}.
\end{align}
As we shall see, replacing our conformal p-values with their oracle counterparts does not change the first-step rejection set $\widehat{\mathcal{R}}_{j\to 0}$. The advantage of working with the oracle p-values is a friendly dependence structure, which ultimately ensures FDR control. The main ideas of the argument are:

*   *Randomization reduction:* for each $j$, $\widehat{\mathcal{R}}_{j\to 0}$ can be expressed as $\widehat{\mathcal{R}}_{j\to 0} = f(p_j,\bm{p}_{-j}^{(j)} )$, where $\bm{p}_{-j}^{(j)}:=\{p_\ell^{(j)}\}_{\ell\neq j}$. For all three pruning options, it holds that $$\textnormal{FDR}\leq \sum_{j=1}^m \mathbb{E}\bigg[\frac{ \mathbb{1}\{  p_j \leq q|f( p_j,\bm{p}_{-j}^{(j)})|/m\}}{ |f( p_j, {\bm{p}}_{-j}^{(j)})| }\bigg].$$
*   *Leave-one-out:* $\widehat{\mathcal{R}}_{j\to 0}$ remains the same if we replace all $\{p_\ell\}$ with $\{\bar{p}_\ell\}$ defined above, i.e., $f( p_j,\bm{p}_{-j}^{(j)})= f(\bar p_j,\bar{\bm{p}}_{-j}^{(j)})$. Since $\widehat{\mathcal{R}}_{j\to 0}$ is obtained by BH, it follows that $f(\bar p_j,\bar{\bm{p}}_{-j}^{(j)})=f(0,\bar{\bm{p}}_{-j}^{(j)})$.
*   *Conditional independence:* under the covariate shift \eqref{eq:cov_shift}, the oracle p-values possess a nice conditional independence structure, namely, $\bar{p}_j \perp\!\!\!\perp \bar{\bm{p}}_{-j}^{(j)} \mid \mathcal{Z}_j$, where $\mathcal{Z}_j = [Z_1,\dots,Z_n,Z_{n+j}]$ is an unordered set of $Z_i=(X_i,Y_i)$, $i=1,\dots,n,n+j$. This, together with the first two steps and the fact that $\bar{p}_j$ is stochastically larger than a uniform random variable conditional on $\mathcal{Z}_j$, gives $\textnormal{FDR}\leq q$.

Along the way, we shall explore interesting connections between Algorithm \ref{alg:bh} and existing ideas in the multiple testing literature.

**Remark.**
Algorithm \ref{alg:bh} is related to the conditional calibration method \cite{fithian2020conditional}, which utilizes sufficient statistics to calibrate a rejection threshold for each individual hypothesis to achieve finite-sample FDR control. Indeed, for each $j$, $s_j:= {q|\widehat{\mathcal{R}}_{j\to 0}  | }/{m}$ can be viewed as the `calibrated threshold' for p-values in their framework; the unordered set $\mathcal{Z}_j$ (after a careful leave-one-out analysis in our proof) plays a similar role as a `sufficient statistic'. Our heterogeneous pruning is similar to their random pruning step, while the other two options generalize their approach. In addition, the procedure and its theoretical analysis are specific to our problem, and they are significantly different.

**Remark.**
Algorithm \ref{alg:bh} is also connected to the eBH procedure \cite{wang2020false}, a generalization of the BH procedure to *e-values*. In the conventional setting with a set of (deterministic) hypotheses $\{H_j\}_{j=1}^m$, e-values are nonnegative random variables $\{e_j\}_{j=1}^m$ such that $\mathbb{E}[e_j]\leq 1$ if $H_j$ is null. For a target level $q\in(0,1)$, the eBH procedure outputs $\mathcal{R}_{\textnormal{eBH}}:=\{j\colon e_j \geq m/(q\widehat{k})\}$, where $\widehat{k} =\max\big\{k\colon \sum_{j=1}^m \mathbb{1}\{e_j\geq m/(qk)\}\geq k\big\}$. One can check that $\mathcal{R}_{\textnormal{dtm}}$ is equivalent to $\mathcal{R}_{\textnormal{eBH}}$ applied to
\begin{align}
\label{eq:eval}
e_j := \frac{\mathbb{1}\{p_j\leq q|\widehat{\mathcal{R}}_{j\to 0}|/m\}}{q|\widehat{\mathcal{R}}_{j\to 0}|/m},
\quad j=1,\dots,m.
\end{align}
Similar to \eqref{eq:general_pvalue}, the $e_j$'s above obey a generalized notion of “null” e-values, in the sense that
\begin{align*}
\mathbb{E}\big[e_j\mathbb{1}\{j\in \mathcal{H}_0\}\big]\leq 1,\quad \textrm{for all}~j=1,\dots,m, 
\end{align*}
see Appendix \ref{app:thm_calib_ite} for details. Furthermore, using the generalized e-values \eqref{eq:eval}, $\mathcal{R}_{\textnormal{hete}}$ and $\mathcal{R}_{\textnormal{homo}}$ are equivalent to running eBH using $\{e_j/\xi_j\}$ and $\{e_j/\xi\}$, respectively. Note that $\{e_j/\xi_j\}$ and $\{e_j/\xi_j\}$ are no longer e-values, yet our procedures still control the FDR while achieving higher power.

Our next result shows that the extra randomness in the second step does not incur too much additional variation for $\mathcal{R}_{\textnormal{hete}}$ and $\mathcal{R}_{\textnormal{homo}}$. The proof of Proposition \ref{prop:asymp_equiv} is in Appendix \ref{app:subsec_asymp_equiv}.

**Proposition 3.2.**
Suppose $\|w\|_\infty \leq M$ for some constant $M>0$. Suppose the distributions of $\{V(X_i,Y_i)\}_{i=1}^n$ and $\{V(X_{n+j},c_{n+j} )\}_{j=1}^m$ have no point mass. Let $\mathcal{R}_{\textnormal{BH}}$ be the rejection set of BH with weighted conformal $p$-values \eqref{eq:def_wcpval_rand}, and $\mathcal{R}$ be any of the three selections from Algorithm \ref{alg:bh}. Let $\mathcal{R}^{(1)}=\{j\colon  p_j\leq q|\widehat{\mathcal{R}}_{j\to 0}|/m\}$ be the first-step selection set. Then the following holds:
1. If $m$ is fixed, then $\lim_{n\to \infty}\mathbb{P}(\mathcal{R}_{\textnormal{BH}}= \mathcal{R} = \mathcal{R}^{(1)})=1$ for each $\mathcal{R} \in \{\mathcal{R}_{\textnormal{homo}},\mathcal{R}_{\textnormal{hete}}, \mathcal{R}_{\textnormal{dtm}}\}$.
2. If $m,n\to \infty$ and the regularity conditions in case (ii) of Theorem \ref{thm:fdr_asymp} hold, then $\frac{|\mathcal{R}^{(1)}\Delta \mathcal{R}_{\textnormal{BH}}|}{|\mathcal{R}^{(1)}|} ~\stackrel{\textnormal{a.s.}}{\to}~0$, $\frac{|\mathcal{R}^{(1)}\Delta \mathcal{R}_{\textnormal{BH}}|}{|\mathcal{R}_{\textnormal{BH}}|} ~\stackrel{\textnormal{a.s.}}{\to}~0$, $ \frac{|\mathcal{R}^{(1)}\Delta \mathcal{R}_{\textnormal{homo}}|}{|\mathcal{R}^{(1)}|} ~\stackrel{\textrm{P}}{\to} ~0$, and $ \frac{|\mathcal{R}_{\textnormal{BH}}\Delta \mathcal{R}_{\textnormal{homo}}|}{|\mathcal{R}_{\textnormal{BH}}|} ~\stackrel{\textrm{P}}{\to} ~0$.
\label{prop:asymp_equiv}

Proposition \ref{prop:asymp_equiv} shows that when the size of the calibration set is sufficiently large, our procedure is close to the BH procedure applied to the weighted conformal p-values, the latter being a deterministic function of the data. In particular, our first-step rejection set is asymptotically equivalent to BH under mild conditions, and the same applies to $\mathcal{R}_{\textnormal{homo}}$. The asymptotic analysis of $\mathcal{R}_{\textnormal{hete}}$ and $\mathcal{R}_{\textnormal{dtm}}$ is challenging. In our numerical experiments, we find that $\mathcal{R}_{\textnormal{hete}}$ is often very close to $\mathcal{R}_{\textnormal{homo}}$, while $\mathcal{R}_{\textnormal{dtm}}$ usually makes too few rejections.

### 3.3 FDR bounds with estimated weights

In practice, the covariate shift $w(\cdot)$ may be unknown. When the conformal p-values are computed with some fitted weights $\widehat{w}(X_i)$, Theorem \ref{thm:est_w} establishes upper bounds on the FDR. The proof is in Appendix \ref{app:thm_est_w}.

**Theorem 3.3.**
Take $w(\cdot):=\widehat{w}(\cdot)$ as the input weight function in Algorithm \ref{alg:bh} and assume $\widehat{w}(\cdot)$ is estimated in a training process with data independent from $\{(X_i,Y_i)\}_{i=1}^{n}\cup\{(X_{n+j},c_{n+j})\}_{j=1}^m$. Under the conditions of Theorem \ref{thm:calib_ite}, for each $\mathcal{R} \in \{\mathcal{R}_{\textnormal{homo}},\mathcal{R}_{\textnormal{hete}}, \mathcal{R}_{\textnormal{dtm}}\}$, we have
\begin{align*}
\textnormal{FDR} \leq  q\cdot \mathbb{E}\bigg[ \frac{\widehat{\gamma}^2}{1+ q(\widehat{\gamma}^2-1)/m} \bigg],
\end{align*}
where $\widehat{\gamma} := \sup_{x\in \mathcal{X}} \max\{  \widehat{w}(x)/w(x),\, w(x)/\widehat{w}(x) \}$.
\label{thm:est_w}

Above, $\widehat{\gamma}\geq 1$ measures the estimation error in $\widehat{w}(\cdot)$ relative to the true weights. When both $w(\cdot)$ and $\widehat{w}(\cdot)$ are bounded away from zero and infinity, $\widehat{\gamma}-1$ is of the same order as $ \sup_x|\widehat{w}(x)-w(x)|$, and hence the FDR inflation in Theorem \ref{thm:calib_ite} converges to zero if $\widehat{w}$ is consistent.

## 4 Application to drug discovery datasets
\label{sec:drug}

As a direct application of WCS, we consider the goal of prioritizing drug candidates. We focus on two tasks: (i) drug property prediction, i.e., selecting molecules that bind to a target protein, and (ii) drug-target interaction prediction, i.e., selecting drug-target pairs with high affinity scores. We use the DeepPurpose library \cite{huang2020deeppurpose} for data pre-processing and model training.

### 4.1 Drug property prediction
\label{subsec:drug_pred}

Our first goal is to find molecules that may bind to a target protein for HIV. Machine learning models are trained on a subset of screened molecules from a drug library, and then used to predict the remaining ones.

We use the HIV screening dataset in the DeepPurpose library with a total size of $n_\textrm{tot}=41127$. In this dataset, the covariate $X\in \mathcal{X}$ is a string that represents the chemical structure of a molecule (encoded by Extended-Connectivity FingerPrints, ECFP), and the response $Y\in \{0,1\}$ is binary, indicating whether the molecule binds to the target protein. Our goal is to select as many new drugs with $Y=1$ as possible while controlling the FDR below a specified level. This can be viewed as the goal \eqref{eq:fdr} with $c_{n+j}\equiv 0.5$.

Oftentimes, experimenters introduce a bias by selecting the first batch of molecules to screen (the training data), which results in a covariate shift between training (calibration) and test data. Here, we mimic an experimentation procedure that builds upon a pre-trained prediction model $\widehat{\mu} \colon \mathcal{X}\to \mathbb{R}$ for binding activity, so that those with higher predicted values are more likely to be included in the training (calibration) data. In our experiment, to reduce computational time, we train a single model for both predicting test samples and for selecting the calibration fold. We take a subset of size $0.4 \times n_{\textrm{tot}}$ as the training set $\mathcal{D}_{\textnormal{train}}$, on which we train a small neural network $\widehat{\mu}(\cdot)$ with three layers trained over three epochs. The remaining $0.6 \times n_{\textrm{tot}}$ samples are randomly selected as the calibration set $\mathcal{D}_{\textnormal{calib}}$ with probability $p(x) = \min\{0.8,  \sigma(\widehat{\mu}(x)-\bar\mu)\}$; here, $x\in \mathcal{X}$, $\bar\mu=\frac{1}{|\mathcal{D}_{\textnormal{train}}|}\sum_{i\in \mathcal{D}_{\textnormal{train}}} \widehat{\mu}(X_i)$ is the average prediction on the training fold, and $\sigma(t) = e^t/(1+e^t)$ is the sigmoid function. The covariate shift \eqref{eq:cov_shift} is thus of the form $w(x)\propto 1/p(x)$, which we assume is known.

We compare the BH procedure with weighted conformal p-values \eqref{eq:def_wcpval_rand}, as well as our Algorithms \ref{alg:bh} and \ref{alg:bh_cond}. The last one is applicable because we here take $c=0.5$ for binary classification. We consider two scores used in BH and Algorithm \ref{alg:bh}:
*   `res`: $V(x,y) = y-\widehat{\mu}(x)$.
*   `clip`: the score $V(x,y) = M\cdot \mathbb{1}\{y>0\} -\widehat{\mu}(x)$, with $M>2\sup_x|\widehat{\mu}(x)|$.

The empirical FDR over $N=200$ independent runs for FDR target levels $q\in \{0.1, 0.2,0.5\}$ is summarized in Figure \ref{fig:drug_pred_fdr}. All algorithms empirically control the FDR below the nominal levels (up to random fluctuation), showing the reliability of WCS in prioritizing drug discovery under covariate shifts. The two scores yield similar power as in Figure \ref{fig:drug_pred_power}.

**Figure 2 Caption:** Empirical FDR for drug property prediction. The label `wBH` is a shorthand for BH, and `wCC.*` for Algorithm \ref{alg:bh} (for `clip` and `res`) or Algorithm \ref{alg:bh_cond} (for `sub`) with three pruning options $*\in \{\texttt{hete},\texttt{homo},\texttt{dtm}\}$. The red dashed lines are the nominal FDR levels.
\label{fig:drug_pred_fdr}

**Figure 3 Caption:** Empirical power for drug property prediction. Everything else is as in Figure \ref{fig:drug_pred_fdr}.
\label{fig:drug_pred_power}

### 4.2 Drug-target interaction prediction
\label{subsec:dti}

We then consider drug-target interaction (DTI) prediction, where the goal is to select drug-target pairs with a high binding score. This task is relevant if a therapeutic company is interested in prioritizing resources for drug candidates that may be effective for any target they are interested in. We use the DAVIS dataset \cite{davis2011comprehensive}, which records real-valued binding affinities for $n_{\textrm{tot}} = 30060$ drug-target pairs. The drugs and the targets are encoded into numeric features using ECFP and Conjoint triad feature (CTF) \cite{shen2007predicting,shao2009predicting}.

We essentially follow the same procedure as in Section \ref{subsec:drug_pred} and only rehearse the main components. First, we randomly sample a subset of size $n_{\textnormal{train}} = 0.2\times n_{\textrm{tot}}$ as $\mathcal{D}_{\textnormal{train}}$, on which we train a regression model $\widehat{\mu}(\cdot)\colon \mathcal{X}\to \mathbb{R}$ using a $3$-layer neural network with $10$ training epochs. This relatively simple model is suitable for numerical experiments on CPUs (one can surely use more complicated prediction models in practice). As before, we use the same model $\widehat{\mu}(\cdot)$ for both predicting test samples and selecting calibration data into first-batch screening (in practice, they can of course be different). We sample a drug-target pair for inclusion in the calibration set with probability $p(x) = \sigma(2\widehat{\mu}(x)-\bar\mu)$, where $\bar\mu$ is the average training prediction as before. Finally, among those not selected in the calibration data, we randomly sample a subset of size $m=5000$ as test data.

We choose a more complicated threshold $c_j$ for the continuous response. For a drug-target pair $X_{n+j}$, we set $c_j$ to be the $q_{\textrm{pop}}$-th quantile of the binding scores of all drug-target pairs in $\mathcal{D}_{\textnormal{train}}$ with the same target. We evaluate $q_{\textrm{pop}} \in\{0.7,0.8\}$. Thus, $c_j:=c(X_{n+j},\mathcal{D}_{\textnormal{train}},q_{\textrm{pop}})$ where $c$ is a mapping that takes both $\mathcal{D}_{\textnormal{train}}$ and the target information in $X_{n+j}$ as inputs. Lastly, we evaluate the BH procedure and Algorithm \ref{alg:bh} with scores:
*   `res`: $V(x,y) = y-\widehat{\mu}(x)$.
*   `clip`: $V(x,y) = M\cdot \mathbb{1}\{y>c(x,\mathcal{D}_{\textnormal{train}},q_{\textrm{pop}})\} + c(x,\mathcal{D}_{\textnormal{train}},q_{\textrm{pop}}) \mathbb{1}\{y\leq c(x,\mathcal{D}_{\textnormal{train}},q_{\textrm{pop}})\}-\widehat{\mu}(x)$ in which $M = 100$.

We always use $\mathcal{D}_{\textnormal{calib}}$ as the calibration set, and set the FDR target as $q\in\{0.1,0.2,0.5\}$.

Figure \ref{fig:drug_dti_fdr} shows false discovery proportions (FDPs) for $q_{\textrm{pop}}=0.8$ in $N=200$ independent. Similar results for $q_{\textrm{pop}}=0.7$ are presented in Appendix \ref{app:subsec_dti}. We see that the FDR is controlled at the desired level for all algorithms and nonconformity scores. This shows the validity of our algorithms and the plausibility of the independence assumptions we make on the drug-target pairs. In this task, we do not observe much difference between deterministic pruning (`WCS.dtm`) and the other two pruning options. Furthermore, we observe that the FDPs across replications tightly concentrate especially for `clip` and $q\in\{0.2,0.5\}$, showing that our algorithms are stable with respect to data splitting (i.e., the randomness in choosing the initial screening sets and in the training process). Comparing the two nonconformity scores, we see that `clip` exploits the error budget and obtains a realized FDR, which is very close to the nominal level, while `res` has a much lower FDR in all cases. Not surprisingly, Figure \ref{fig:drug_dti_power} shows that `clip` has much higher power.

**Figure 4 Caption:** Empirical FDR for drug-target interaction prediction with $q_{\textrm{pop}}=0.8$. The shorthand `WBH` stands for BH procedure, and `WCS.*` for Algorithm \ref{alg:bh} with three pruning options $*\in \{\texttt{hete},\texttt{homo},\texttt{dtm}\}$. The red dashed lines are the nominal FDR levels. Solid lines are empirical averages (here empirical FDR).
\label{fig:drug_dti_fdr}

**Figure 5 Caption:** Empirical power for drug-target interaction prediction with $q_{\textrm{pop}}=0.8$. Everything else is as in Figure \ref{fig:drug_dti_fdr}.
\label{fig:drug_dti_power}

## 5 Application to multiple individual treatment effects
\label{sec:ite}

We now apply our method to infer individual treatment effects (ITEs) \cite{lei2021conformal,jin2023sensitivity} under the potential outcomes framework \cite{imbens2015causal}. The ITE describes the difference between an individual's outcomes when receiving a treatment versus not; its variation comes from both individual characteristics and intrinsic uncertainty in the outcomes. Inference on multiple ITEs is useful in assisting reliable personalized decision making.

We consider a super-population setting, where $(X_i,O_i(1),O_i(0),T_i)$ are drawn i.i.d. from a joint distribution $\mathcal{P}$ (distinct from the distributions $P$ and $Q$ we use for the data). Here, $X_i\in \mathcal{X}$ is the features, $T_i\in \{0,1\}$ indicates whether unit $i$ receives treatment, and $O_i(1),O_i(0)$ are the potential outcomes under treatment and not, respectively. Under the standard SUTVA \cite{imbens2015causal}, we observe $(X_i,O_i,T_i)$, where $O_i=O_i(T_i)=T_iO_i(1)+(1-T_i)O_i(0)$. We will specify how the treatment $T_i$ is allocated later on. We focus on units *in the study*, i.e., those who have received a certain treatment option. The ITE of unit $i$ is the random variable $\Delta_i = O_i(1)-O_i(0)$. As only one potential outcome is observed, a crucial part of inferring the ITE is to predict the counterfactual, i.e., $O_i(1)$ for those units with $T_i=0$, and $O_i(0)$ for $T_i=1$. In the following, we consider simultaneous inference for the ITEs of a set of units in the control group (inference for treated units is similar).

Formally, the test samples are $\{ X_{n+j} \}_{j=1}^m$ are units in the control group (the outcome $O_{n+j}$ is observed). Our goal is to screen for positive ITEs with FDR control, i.e., finding a subset $\mathcal{R}\subseteq\{1,\dots,m\}$, such that
\begin{align}
\label{eq:fdr_ite}
\textnormal{FDR} := \mathbb{E}\Bigg[  \frac{\sum_{j=1}^m \mathbb{1}\{O_{n+j}(1)\leq O_{n+j}(0),~j\in \mathcal{R}\}}{1\vee |\mathcal{R}|}   \Bigg]  \leq q.
\end{align}

To cast the counterfactual problem in our framework, set $Y_{n+j}:= O_{n+j}(1)$ to be the unobserved outcome, and the thresholds $c_{n+j} := O_{n+j}(0)$ to be the observed outcomes. To infer $Y_{n+j}$, the training data is $\{(X_i,Y_i )\}_{i=1}^n = \{(X_i,O_i(1))\}_{i=1}^n$ for which $T_i=1$. Conditional on the treatment status, the training samples are i.i.d. from $P:=\mathcal{P}_{X,O(1)\mid T=1}$, while the test samples $\{(X_{n+j},Y_{n+j})\}_{j=1}^m$ are i.i.d. from $Q:=\mathcal{P}_{X,O(1)\mid T=0}$. The relationship between $P$ and $Q$ will depend on the treatment assignment mechanism.

### 5.1 Warm-up: completely randomized experiments

As a warmup, consider a completely randomized experiment, where the treatment assignments $T_i$ are i.i.d. draws from $\textrm{Bern}(\pi)$ for some $\pi\in(0,1)$ independently from everything else. In this case, $P =Q = \mathcal{P}_{X,O(1)}$ and it suffices to use conformal p-values without weights. The testing procedure for randomized experiments has already been studied in \cite{jin2022selection}. Take any *monotone* nonconformity score $V(\cdot,\cdot)$ obtained from an independent training process, and compute $V_i = V(X_i,O_i) = V(X_i,O_i(1))$ for $i=1,\dots,n$, and $\widehat{V}_{n+j} =V(X_{n+j},O_{n+j})= V(X_{n+j},O_{n+j}(0))$ for $j=1,\dots,m$. Then construct conformal p-values \eqref{eq:weighted_pval} with $w(\cdot)\equiv 1$, and run the BH procedure with these p-values at level $q$. FDR control is a corollary of \cite[Theorem 2.3]{jin2022selection}, and we omit the proof.

**Corollary 5.1.**
The procedure above has FDR \eqref{eq:fdr_ite} at most $q$.
\label{cor:ite_randomized}

### 5.2 Stratified randomization and observational studies

We now consider a more general setting where the treatment assignment may depend on the observed covariates, formalized as the following strong ignorability condition \cite{imbens2015causal}.

**Assumption 5.2 (Strong ignorability).**
Under the joint distribution $\mathcal{P}$, it holds that $(O(1),O(0)) \perp\!\!\!\perp T \mid X$. Equivalently, the treatment assignments are independently generated from $T_i\sim \textrm{Bern}(e(X_i))$, where $e\colon \mathcal{X}\to (0,1)$ is known as the propensity score.
\label{assump:ignor}

Assumption \ref{assump:ignor} is automatically satisfied in stratified randomization experiments, where the treatment is randomized in a way that only depends on the covariates, and the propensity score $e(\cdot)$ is known. In observational studies where the treatment assignment mechanism is completely unknown, Assumption \ref{assump:ignor} is standard for the identifiability of average treatment effects \cite{rosenbaum2002observational}. Inference on ITEs when this assumption is violated (i.e., when there is unmeasured confounding) is studied in \cite{jin2023sensitivity}; that said, multiple testing under confounding needs additional techniques, and is beyond the scope of the current work.

Under Assumption \ref{assump:ignor}, the covariate shift condition \eqref{eq:cov_shift} holds \cite{lei2021conformal}, as
\begin{align*}
\frac{\mathrm{d} Q}{\mathrm{d} P}(x,y) = \frac{\mathrm{d} \mathcal{P}_{X,O(1)\mid T=0}}{\mathrm{d} \mathcal{P}_{X,O(0)\mid T=1}}(x,y) = \frac{\mathrm{d} \mathcal{P}_{X\mid T=0}}{\mathrm{d} \mathcal{P}_{X \mid T=1}}(x) = w(x) := \frac{\pi(1-e(x))}{(1-\pi) e(x)},
\end{align*}
where $e(x)=\mathcal{P}(T=1\mid X=x)$ is the propensity score, and $\pi=\mathcal{P}(T=1)$ is the marginal probability of being treated. As such, Algorithm \ref{alg:bh} is readily applicable when $e(\cdot)$ is known.

**Corollary 5.3.**
Suppose $e(\cdot)$ is known, and Assumption \ref{assump:ignor} holds. Consider calibration data $\{(X_i,O_i(1)\colon T_i=1\}_{i=1}^n$, test data $\{X_{n+j}\colon T_{n+j}=1\}_{j=1}^m$, thresholds $\{O_{n+j}(0)\colon T_{n+j}=0\}_{j=1}^m$, any monotone score $V$, and weight function $w(x) \propto (1-e(x))/{e(x)}$ as the input of Algorithm \ref{alg:bh}. Then any selection procedure $\mathcal{R} \in \{\mathcal{R}_{\textnormal{hete}}, \mathcal{R}_{\textnormal{homo}}, \mathcal{R}_{\textnormal{dtm}}\}$ has FDR at most $q$.
\label{cor:ite_weight}

The propensity score function $e(\cdot)$ is unknown for observational data. Under Assumption \ref{assump:ignor}, we can estimate the propensity scores (hence the weight function) using an independent training fold and plug this estimate into the construction of p-values. As a corollary of Theorem \ref{thm:est_w}, we can develop an FDR bound for observational data.

**Corollary 5.4.**
In the setting of Corollary \ref{cor:ite_weight}, take $w(x) \propto (1-\widehat{e}(x))/{\widehat{e}(x)}$ as the input of Algorithm \ref{alg:bh}, where $\widehat{e}(\cdot)$ is an estimate of $e(\cdot)$ which is independent of the calibration and training data. Then any selection procedure $\mathcal{R} \in \{\mathcal{R}_{\textnormal{hete}}, \mathcal{R}_{\textnormal{homo}}, \mathcal{R}_{\textnormal{dtm}}\}$ obeys
\begin{align*}
\textnormal{FDR} \leq  q\cdot \mathbb{E}\bigg[ \frac{\widehat{\gamma}^2}{1+ q(\widehat{\gamma}^2-1)/m} \bigg],
\qquad 
\widehat{\gamma} := \sup_{x\in\mathcal{X}}\max\bigg\{ \frac{1-e(x)\widehat{e}(x)}{e(x)(1-\widehat{e}(x))},
\frac{e(x)(1-\widehat{e}(x))}{1-e(x)\widehat{e}(x)}  \bigg\}.
\end{align*}

At a high level, the conformal p-values compare the observed outcome $O_{n+j}(0)$ of the control units to the empirical distribution of $O_{i}(1)$ in the calibration data; the latter—after proper weighting—shows the typical behavior of the counterfactual $O_{n+j}(1)$, and provides evidence for whether it may be larger than $O_{n+j}(0)$, i.e., whether $\Delta_{n+j}$ is positive.

In predicting $O_{n+j}(1)$, we use the marginal distribution of $(X,O(1))$ from the calibration data, but ignore the information in the observed outcome $O(0)$. Put differently, our inference for ITEs is valid regardless of how the potential outcomes are coupled. Thus, the actual false discovery rate may be lower than $q$. Indeed, it is observed in \cite{jin2023sensitivity} that the FDR for identifying positive ITEs—even without adjusting for multiplicity—can sometimes be lower than the nominal level. Next, we are to empirically investigate the FDR and power of our method under various couplings of potential outcomes. A theoretical understanding is left for future work.

### 5.3 Simulation studies

We design joint distributions of the variables $(X,O(1),O(0),T)$ with various covariate distributions, regression functions, and couplings of potential outcomes. The covariates $X_i \in \mathbb{R}^{10}$ ($p = 10$) are obtained via $X_{ij}=\Phi(X_{ij}^0)$, $j=1,\dots,10$, where $X_{i}^0\in \mathbb{R}^{10}$ are i.i.d. $\mathcal{N}(0,\Sigma)$, and $\Phi(\cdot)$ is the CDF of a $\mathcal{N}(0,1)$ random variable. We set $\Sigma=I_{p}$ for the independent case and $\Sigma_{k,j}=0.9^{|k-j|}$ for the correlated case. We consider three cases:
*   Setting 1: $O_i(0) = 0.1\, \epsilon_{0,i}$, $O_i(1) = \max\{ 0, \mu_1(X_i) + \sigma_1(X_i) \, \epsilon_{1,i} \}$, where $\sigma_1 (x) =   0.2 -  \log x_1$ and $ \mu_1(x) = {4}/{(1+e^{-12x_1-0.5})(1+e^{-12x_2-0.5})}$.
*   Setting 2 is a mixture of deterministic and stochastic ITEs. With probability $0.1$, set $O_i(1)= 0.1\, \epsilon_{0,i}-0.5$, and $O_i(0)=O_i(1)+0.05$; otherwise, set them as in Setting 1.
*   Setting 3: $O_i(0) = \mu_0(X_i) + 0.1 \, \epsilon_{0,i}$, $O_i(1)=\max\{0, \mu_1(X_i) + \sigma_1(X_i) \, \epsilon_{1,0}\}$, where $\sigma_1(\cdot)$ is as in Setting 1, and $\mu_0(x) = {2}/{(1+e^{-3x_1-0.5})(1+e^{-3x_2-0.5})}$, $\mu_1(x)=0.1+1.5\, \mu_0(x)$.

Above, the variables ($\epsilon_{0,i},\epsilon_{1,i}$) are i.i.d. with $\mathcal{N}(0,1)$ as marginals. We consider three coupling scenarios: (i) independent, in which $\epsilon_{0,i}$ and $\epsilon_{1,i}$ are independent; (ii) negative, in which $\epsilon_{1,i}=-\epsilon_{0,i}$; (iii) positive, in which $\epsilon_{1,i} =\epsilon_{0,i}$. We generate treatment indicators via $T_i\sim \textrm{Bern}(e(X_i))$ independently, where $e(x)= (1+\textrm{Beta}_{2,4}(x_1))/{4}$, and $\textrm{Beta}_{2,4}(\cdot)$ is the CDF of the Beta distribution with shape parameters $(2,4)$. Finally, we set those $T_i=1$ as the calibration data ($n=250$), and those $T_i=0$ as the test sample ($m=100$); as training data we select $n_\textnormal{train}=750$ units, which are used to construct four conformity scores:
*   `reg`: $V(x,y) = y-\widehat{\mu}(x)$, where $\widehat{\mu}(x)$ is an estimate of $\mu_1(x)$ using regression forest from the `grf` R-package \cite{athey2019generalized}.
*   `cdf`: $V(x,y) = \widehat{F}(x,y)$ \cite{chernozhukov2021distributional}, where $\widehat{F}(x,y)$ is an estimate of $\mathcal{P}(O(1)\leq y\mid X=x)$. Here, we fit $\widehat{F}$ via inverting quantile regression forests from `grf`.
*   `oracle`: $V(x,y) = \mathcal{P}(O(1)\leq y\mid X=x)$, which is not computable from data and is shown only for illustration.
*   `cqr`: $V(x,y) =  y -\widehat{q}_\beta(x)$ \cite{romano2019conformalized}, where $\widehat{q}_\beta(x)$ is an estimate of the $\beta$-th quantile of $\mathcal{P}_{O(1)\mid X=x}$, $\beta\in\{0.2,0.5,0.8\}$, from the quantile regression forests in `grf`.

After generating the three folds of data, we fit $V(\cdot,\cdot)$ on the training fold, and run four procedures on the calibration and training folds: BH with weighted conformal p-values in \eqref{eq:def_wcpval_rand} (named `WBH` in the plot), and Algorithm \ref{alg:bh} with three pruning options (named `WCS.hete`, `WCS.homo`, `WCS.dtm`, respectively). The experiment is repeated for $N=1000$ independent runs.

Among the three, setting 2 is the most challenging: $10\%$ of the samples are with slightly negative ITEs; they are likely to be falsely rejected since the selection procedure ignores the coupling (it may reject a sample with a small value of $O(0)$ even if $O(1)$ is smaller). We expect setting 1 to be the least challenging as there is a strong signal in $O(1)$. Setting 3 is more challenging than setting 1 since $O(1)$ and $O(0)$ are close to each other as the regression functions are similar.

**Figure 6 Caption:** Empirical FDR for finding positive ITEs. The columns represent three settings with either independent (`ind`) or correlated (`corr`) covariates. Each row corresponds to a coupling of potential outcomes (positive, negative, and no coupling). In each subplot, the $y$-axis is the nonconformity score, and the $x$-axis is the empirical FDR. The red dashed lines indicate the nominal level $q=0.10$.
\label{fig:ite_simu_fdr}

**FDR control.**
The empirical FDR in all settings is plotted in Figure \ref{fig:ite_simu_fdr}. We observe FDR control for all methods, including the BH procedure. (We omit the results of $\mathcal{R}_{\textnormal{dtm}}$ because it does not succeed in making any selection.) Across settings with the same regression function, those with correlated covariates often have higher FDR. The realized FDR also varies with the scores: `oracle` incurs very low FDR, while its empirical counterpart `cdf` has higher FDR. On the other hand, quantile regression based scores are robust to the quantile level $\beta$ in the sense that `cqr` achieves similar FDR for all choices of $\beta$.

The impact of coupling on FDR is less consistent across settings. The positive coupling from setting 1 yields a low FDR, perhaps because it is difficult to obtain sufficiently strong evidence which leads to fewer rejections (see the power analysis below). The three couplings yield similar FDRs for setting 2. For setting 3, positive coupling leads to the highest FDR.

**Power.** The empirical power is defined as
$
\textrm{Power} := 
\mathbb{E}\Big[\frac{\sum_{j=1}^m\mathbb{1}\{j\in \mathcal{R},O_{n+j}(1)>O_{n+j}(0)\}}{\sum_{j=1}^m \mathbb{1}\{O_{n+j}(1)>O_{n+j}(0)\}}\Big].
$
As we see in Figure \ref{fig:ite_simu_power}, the power is in general higher when the covariates are correlated. The ITE depends on $(X_1,X_2)$, and in our design, higher correlation between the first two entries leads to higher power. The power also varies with the nonconformity scores. The regression-based `reg` leads to lower power than other methods. This means that regression functions may be underpowered in capturing the essential information for contrasting the distributions of two potential outcomes. Quantile-regression based methods (`cqr`) achieve similar power for different choices of $\beta$ in all settings. Finally, fitting the conditional distribution function (`cdf`) is the most powerful option in settings 1 and 3, but is far less powerful in setting 2. Its oracle version (`oracle`) is also surprisingly less powerful; we observe that the oracle cdf cannot create sufficiently small p-values, while the estimated one is able to do so due to fluctuations caused by estimation uncertainty.

**Figure 7 Caption:** Empirical power for finding positive ITEs. Everything else is as in Figure \ref{fig:ite_simu_fdr}.
\label{fig:ite_simu_power}

The impact of coupling on the power is consistent across settings. In general, negative coupling leads to the highest power, while positive coupling leads to the lowest. This is because the conformal p-values compare the observed outcome to the *marginal* distribution of the counterfactuals, without accounting for their joint distribution. Thus, when $O_{n+j}(0)$ is extremely small, i.e., when we see a small p-value and $j\in \mathcal{R}$, under negative coupling, it is more probable that $O_{n+j}(1)$ is relatively large, hence leading to a true discovery. On the contrary, under positive coupling, a large $O_{n+j}(1)$ often corresponds to a relatively large $O_{n+j}(0)$ and hence a large conformal p-value, which may not be selected.

Finally, $\mathcal{R}_{\textnormal{dtm}}$ makes no selection. This may be due to the threshold effect: since the selection requires $p_j\leq q|\widehat{\mathcal{R}}_{j\to 0}|/m$ for test samples $j$ with the smallest $|\widehat{\mathcal{R}}_{j\to 0}|$, the pruning step may exclude too many candidates. We recommend using $\mathcal{R}_{\textnormal{hete}}$ and $\mathcal{R}_{\textnormal{homo}}$ in practice.

**Stability.** As aforementioned in Section \ref{sec:method}, our method introduces extra randomness. It is shown in Proposition \ref{prop:asymp_equiv} that this becomes asymptotically negligible, so that our selection set is asymptotically the same as that the BH procedure yields. We empirically evaluate the discrepancy in the selections, namely, $|\mathcal{R}\Delta \mathcal{R}_{\textnormal{BH}}|/|\mathcal{R}_{\textnormal{BH}}|$, for $\mathcal{R}$ being either $\mathcal{R}_{\textnormal{homo}}$ or $\mathcal{R}_{\textnormal{hete}}$. The results are shown in Figure \ref{fig:ite_simu_dif}, confirming our theory. Though we were not able to prove the asymptotic equivalence of $\mathcal{R}_{\textnormal{hete}}$ and $\mathcal{R}_{\textnormal{BH}}$ in the $m,n\to \infty$ regime, the discrepancy is still small (despite being larger than that of $\mathcal{R}_{\textnormal{homo}}$). All in all, the impact of additional randomness seems acceptable in practice.

**Figure 8 Caption:** Discrepancy between selection rules $|\mathcal{R}\Delta \mathcal{R}_{\textnormal{BH}}|/|\mathcal{R}_{\textnormal{BH}}|$. Everything else is as in Figure \ref{fig:ite_simu_fdr}.
\label{fig:ite_simu_dif}

The discrepancy for $\mathcal{R}_{\textnormal{dtm}}$ from $\mathcal{R}_{\textnormal{BH}}$ is large in general is $\mathcal{R}_{\textnormal{dtm}}$ is usually an empty set, and hence we did not plot it. We conjecture that the stability of $\mathcal{R}_{\textnormal{dtm}}$ claimed in case (i) from Proposition \ref{prop:asymp_equiv} may not necessarily apply to the moderately large sample sizes $m=100$ and $n=250$, and $\mathcal{R}_{\textnormal{dtm}}$ may be far from $\mathcal{R}_{\textnormal{BH}}$ for such sample sizes. In such cases, we recommend using $\mathcal{R}_{\textnormal{homo}}$ or $\mathcal{R}_{\textnormal{hete}}$ for better stability and power.

### 5.4 Real data analysis

We revisit the NSLM observational dataset from \cite{carvalho2019assessing} and use our method to detect multiple positive individual treatment effects. It is a semi-synthetic observational dataset curated from a real randomized experiment, also analyzed in \cite{lei2021conformal,jin2023sensitivity}.

Figure 2 in \cite{jin2023sensitivity} plots the $\Gamma$-value, a measure of robustness of positive ITEs against unmeasured confounding, versus individual covariates. While units with larger $\Gamma$-values are of natural interest, properties of inference hold on average, instead of over *selected* units (e.g. those with high $\Gamma$-values). We are to produce a similar plot (Figure \ref{fig:ite_pval}) for detecting positive ITEs, with guarantees over units that exhibit the strongest evidence. (We do not consider the confounding issue, which may need nontrivial extension of our techniques.)

We randomly subsample three disjoint folds of size $5000$, $1000$, and $391$ from the original dataset. The first is the training fold, and consists of both treated and control units. The calibration data consists of all the $n = 997$ treated units in the second fold. The test data consists of all the $m = 256$ control units in the last fold. To deploy our procedure, we first train a propensity score model $\widehat{e}(\cdot)$ using a regression forest from the `grf` R-package, and set $\widehat{w}(x) = (1-\widehat{e}(x))/{\widehat{e}(x)}$. We then use a quantile regression forest from `grf` on the training data to fit $\widehat{q}_{0.5}(x)$, the conditional median of $O(1) \mid X$, and set $V(x,y) = y - \widehat{q}_{0.5}(x)$. Finally, we apply Algorithm \ref{alg:bh} as in Theorem \ref{thm:calib_ite} to test for positive ITEs.

Figure \ref{fig:ite_cdf} plots the empirical CDF of the weighted conformal p-values \eqref{eq:weighted_pval} computed with $c_{n+j} := O_{n+j}(0)$ on the test samples. These p-values are stochastically smaller than Unif([0,1]), showing evidence for positive treatment effects in general. Yet, to identify positive ITEs with rigorous error control we must apply multiple testing ideas here.

**Figure 9 Caption:** Empirical CDF of the weighted conformal p-values for detecting positive ITEs.
\label{fig:ite_cdf}

We observe $\mathcal{R}_{\textnormal{BH}} = \mathcal{R}_{\textnormal{homo}} = \mathcal{R}_{\textnormal{hete}}$ while $\mathcal{R}_{\textnormal{dtm}}=\varnothing$ at FDR levels $q\in\{0.1,0.2,0.5\}$. Figure \ref{fig:ite_pval} plots the weighted conformal p-values versus school achievement levels of test units (each dot represents a test unit) with red dots (from light to dark) identified as positive ITEs at various FDR levels. Students in schools with moderate achievement levels demonstrate the strongest evidence of benefiting from the treatment.

**Figure 10 Caption:** P-values and rejection sets for detecting positive ITEs among control units.
\label{fig:ite_pval}

## 6 Application to outlier detection
\label{sec:outlier}

Finally, we study the extension of our framework for outlier detection, where the calibration inliers may follow a distinct distribution compared with test inliers. We discuss a new setup and apply a variant of Algorithm \ref{alg:bh} to bank marketing data.

### 6.1 Hypothesis-conditional FDR control
\label{subsec:hypo_cond_fdr}

Assuming access to i.i.d. training data $\{  Z_i  \}_{i=1}^n $, we consider a set of test data $\{ Z_{n+j}  \}_{j=1}^m $ for which the $Z_{n+j}$'s may only be partially observed (e.g., if $Z=(X,Y)$ we would observe the features $X$ but not the response $Y$). We are interested in some null hypotheses $\{H_j \}_{j=1}^m $ associated with the test samples. As before, whether $H_j$ is true can be a random event depending on $Z_{n+j}$; this includes our previous problem with $H_j\colon Y_{n+j}\leq c_{n+j}$.

#### 6.1.1 Hypothesis-conditional covariate shift
\label{subsec:outlier}

**Assumption 6.1 (Hypothesis-conditional covariate shift).**
$\{Z_i\}_{i=1}^n\cup\{Z_{n+j}\}_{j=1}^m$ are mutually independent. Also, conditional on the subset $\mathcal{H}_0\subset\{1,\dots,m\}$ of all null hypotheses, it holds that $\{Z_i\}_{i=1}^n\stackrel{\textnormal{i.i.d.}}{\sim} P$, and $\{Z_{n+j}\}_{j\in \mathcal{H}_0} \stackrel{\textnormal{i.i.d.}}{\sim} Q$, where $ \mathrm{d} Q/\mathrm{d} P (Z)=w(X)$ for some function $w\colon \mathcal{X}\to \mathbb{R}^+$ and $X\subseteq Z$.
\label{assump:label_conditional}

A special case of problem \eqref{eq:fdr} obeys these assumptions.

**Example 6.2 (Binary classification).**
Set $Z=(X,Y)$. where $Y\in\{0,1\}$ is a binary response. Suppose the goal is to find positive $Y_{n+j}$, e.g. an active drug or a qualified candidate. We only observe the covariates $\{X_{n+j}\}_{j=1}^m$ for the test samples $\{(X_{n+j},Y_{n+j})\}_{j=1}^m\stackrel{\textnormal{i.i.d.}}{\sim} Q$. Consider a reference dataset that only preserves positive samples among a set of i.i.d. data from a covariate shifted distribution $P$ ; that is, $\mathcal{D}_{\textnormal{calib}} = \{Z_i\colon Y_{i}=0\}$ where $\{(X_i,Y_i)\}\stackrel{\textnormal{i.i.d.}}{\sim} P$. Although the (super-population) covariate shift \eqref{eq:cov_shift} no longer applies, conditional on $\mathcal{H}_0 = \{j\colon Y_{n+j}=0\}$, it still holds that $Z_i\stackrel{\textnormal{i.i.d.}}{\sim} P_{Z\mid Y=0}$ for $Z_i\in \mathcal{D}_{\textnormal{calib}}$, and $Z_{n+j}\stackrel{\textnormal{i.i.d.}}{\sim} Q_{Z\mid Y=0}$ for $j\in \mathcal{H}_0$.
\label{ex:binary}

The above example also applies to candidate screening with constant thresholds $c_{n+j} \equiv c_0$. Setting $\widetilde{Y} = \mathbb{1}\{Y > c_0\}\in\{0,1\}$ and $\mathcal{D}_{\textnormal{calib}} = \{Z_i\colon Y_i\leq c_0\}$, all arguments apply similarly to $\widetilde{Z}=(X,\widetilde{Y})$. However, this does not necessarily apply when the thresholds $c_{n+j}$ are random variables (especially when no 'threshold $c_j$' is observed in the calibration data, such as in the counterfactual inference problem we will study later).

Another application is outlier detection under covariate shifts.

**Example 6.3 (Outlier detection).**
We revisit outlier detection \cite{bates2021testing} while allowing for identifiable covariate shifts between the calibration inliers and the test inliers. Given $\{Z_i\}_{i=1}^n$ drawn i.i.d. from an unknown distribution $P$ and a set of test data $\{Z_{n+j}\}_{j=1}^m$, we assume the inliers in the test data are i.i.d. from a distribution $Q$ with $\mathrm{d} Q/\mathrm{d} P(Z)=w(Z)$ for a known function $w$, while allowing outliers to be from arbitrary distributions. The covariate shift may happen, for example, when inliers *were* from $Q$ but the calibration set is selected with preferences relying on $z$: for instance, one may include more female users to balance the gender distribution when curating a reference panel of normal transactions (inliers). In this case, $\mathcal{H}_0 = \{j\colon Z_{n+j}\sim Q\}$ is a deterministic set, and Assumption \ref{assump:label_conditional} clearly holds.

The outlier detection example is closely related to identifying concept drifts.

**Example 6.4 (Concept shift detection).**
Letting $Z=(X,Y)$ where $X\in \mathcal{X}$ is the family of covariates and $Y$ is the response, concept shift focuses on potential changes in the conditional distribution of $Y$ given $X$. Given calibration data $\{Z_i\}_{i=1}^n\stackrel{\textnormal{i.i.d.}}{\sim} P$ and independent test data $\{Z_{n+j}\}_{j=1}^m$, \cite{hu2020distribution} assume $\{Z_{n+j}\}_{j=1}^m \stackrel{\textnormal{i.i.d.}}{\sim} Q$, and test for the global null $H_0\colon \mathrm{d} Q/\mathrm{d} P(Z) = w(X)$ for some $w\colon \mathcal{X}\to \mathbb{R}^+$. They achieve this by combining independent p-values after sample splitting. Our framework can be used to test individual concept drifts with dependent p-values. For instance, assume $\{X_{n+j}\}_{j=1}^m\stackrel{\textnormal{i.i.d.}}{\sim} Q_X$ for some unknown (but estimable) $Q_X$, we can test whether $P_{Y_{n+j}\mid X_{n+j}} = P_{Y\mid X}$. The null hypotheses can be formulated as $H_j\colon Z_{n+j}\sim Q$, where $\mathrm{d} Q/\mathrm{d} P(Z)=w(X)$ for some $w\colon \mathcal{X}\to \mathbb{R}^+$ that is either known or can be estimated well under proper conditions.

#### 6.1.2 Multiple testing procedure

Our procedure for outlier detection under covariate shift is detailed in Algorithm \ref{alg:bh_cond}. This slightly modifies Algorithm \ref{alg:bh} by removing the thresholds (note differences in lines 1, 2, and 4). In the classification or constant threshold problem (Example \ref{ex:binary}), it suffices to set $Z=X$ and leave out $Y$.

**Algorithm 2: Hypothesis-conditional Weighted Conformalized Selection**
\label{alg:bh_cond}
**Input:** Calibration data $\{Z_i\}_{i=1}^n$, test data  $\{Z_{n+j}\}_{j=1}^m$, weight function $w(\cdot)$, FDR target $q\in(0,1)$, monotone nonconformity score $V\colon \mathcal{X}\times\mathcal{Y}\to \mathbb{R}$, pruning method $\in\{\texttt{hete}, \texttt{homo}, \texttt{dtm}\}$.

1. Compute $V_i = V(Z_i)$ for $i=1,\dots,n$, and $ {V}_{n+j}= V(Z_{n+j})$ for $j=1,\dots,m$.
2. Construct weighted conformal p-values $\{ p_j\}_{j=1}^m$ as in \eqref{eq:weighted_pval} with $\widehat{V}_{n+j}$ replaced by $V_{n+j}$.

*-- First-step selection --*
3. **for** $j=1,\dots,m$ **do**
4. Compute p-values $\{ {p}_\ell^{(j)}\}$ as in \eqref{eq:mod_pval} with $\widehat{V}_{n+\ell}$ replaced by $V_{n+\ell}$ for all $\ell=1,\dots,m$.
5. (BH procedure) Compute $k^*_j = \max\big\{k\colon 1 +\sum_{\ell\neq j} \mathbb{1}\{{p}_\ell^{(j)}\leq qk/m\}\geq k\big\}$.
6. Compute $\widehat{\mathcal{R}}_{j\to 0} = \{j\}\cup\{\ell \neq j\colon  {p}_\ell^{(j)}\leq q k^*_j /m\}$.
7. **end for**
8. Compute the first-step selection set $\mathcal{R}^{(1)} = \{j\colon  {p}_j \leq q|\widehat{\mathcal{R}}_{j\to 0}|/m\}$.

*-- Second-step selection --*
9. Compute $\mathcal{R} = \mathcal{R}_{\textrm{hete}}$ or $\mathcal{R} = \mathcal{R}_{\textrm{homo}}$ or $\mathcal{R} = \mathcal{R}_{\textrm{dtm}}$ as in Algorithm \ref{alg:bh}.

**Output:** Selection set $\mathcal{R}$.

Algorithm \ref{alg:bh_cond} returns to the conventional perspective, where the null set is deterministic, and the null p-values are dominated by Unif$([0,1])$. That is, for $p_j$ constructed in Line 2 of Algorithm \ref{alg:bh_cond}, it holds that
\begin{align*}
\mathbb{P}(p_j \leq t \mid j\in \mathcal{H}_0 ) \leq t\quad \textrm{for all }t\in[0,1]. 
\end{align*}
After conditioning on $\mathcal{H}_0$, we no longer need to deal with the randomness of the hypotheses and their interaction with the p-values. The only issue is the mutual dependence among the p-values, which is addressed using a similar idea as in our theoretical analysis of Algorithm \ref{alg:bh}.

Using calibration data obeying the covariate shift assumption, Algorithm \ref{alg:bh_cond} achieves a slightly stronger hypotheses-conditional FDR control. The proof of Theorem \ref{thm:fdr_cond} is in Appendix \ref{app:thm_outlier}.

**Theorem 6.5.**
Under Assumption \ref{assump:label_conditional}, Algorithm \ref{alg:bh_cond} yields
\begin{align*}
\mathbb{E}\bigg[  \frac{ |\mathcal{R}\cap \mathcal{H}_0| }{1\vee |\mathcal{R}|} \Biggm \mathcal{H}_0 \bigg]   \leq q\cdot \frac{|\mathcal{H}_0|}{m}
\end{align*}
for any fixed $q\in(0,1)$, and each $\mathcal{R} \in \{\mathcal{R}_{\textnormal{homo}},\mathcal{R}_{\textnormal{hete}}, \mathcal{R}_{\textnormal{dtm}}\}$.
\label{thm:fdr_cond}

#### 6.1.3 Comparison with Algorithm \ref{alg:bh}
\label{subsubsec:compare}

In binary classification, or more generally, WCS with a constant threshold, we have shown in Example \ref{ex:binary} that Algorithm \ref{alg:bh_cond} yields FDR control. In this case, Algorithms \ref{alg:bh} and \ref{alg:bh_cond} differ in terms of (i) power, and (ii) distributional assumptions. We elaborate on these distinctions.

First, Algorithm \ref{alg:bh_cond} only uses a subset of calibration data to construct p-values, which leads to a power loss for specific choices of nonconformity scores; see \cite[Appendix A.1]{jin2022selection} for the i.i.d. case.
Let us consider the binary setting. Suppose we have access to a set of calibration data $\{(X_i,Y_i)\}$ consisting of both $Y=1$ and $Y=0$ samples. As discussed in Example \ref{ex:binary}, Assumption \ref{alg:bh_cond} holds if we only use data in the subset $\mathcal{I}_0 = \{i\colon Y_i=0\}$ as $\mathcal{D}_{\textnormal{calib}}$ in Algorithm \ref{alg:bh_cond}. In contrast, Algorithm \ref{alg:bh} uses all data points. Suppose we set $V(x,y) = My - \widehat{\mu}(x)$ and $c_{n+j} \equiv 0$ in Algorithm \ref{alg:bh}, where $\widehat{\mu}(\cdot)$ is a fitted point prediction, and $M>2\sup_{x\in\mathcal{X}}|\widehat{\mu}(x)|$ is a sufficiently large constant. Similarly, we set $V(x)=M-\widehat{\mu}(x)$ in Algorithm \ref{alg:bh_cond}. This construction ensures
\begin{align}
\label{eq:monotone_V}
\inf_{x\in \mathcal{X}} V(x,1) = M-\sup_{x\in \mathcal{X}}\widehat{\mu}(x) 
> \sup_{x\in \mathcal{X}}|\widehat{\mu}(x)| \geq \sup_{x\in \mathcal{X}} V(x,0).
\end{align}
Theorems \ref{thm:calib_ite} and \ref{thm:fdr_cond} state that the FDR is controlled for both approaches. However, letting $p_j$ denote the p-values constructed in Algorithm \ref{alg:bh} and $p_j'$ denote those in Algorithm \ref{alg:bh_cond}, we note that
\begin{align*}
p_j &= 
\frac{\sum_{i\in \mathcal{I}_0} w(X_i)\mathbb{1}{\{V(X_i,0) < V(X_{n+j},0) \}} 
+  w(X_{n+j}) }{\sum_{i=1}^n w(X_i) + w(X_{n+j})} 
+ \frac{ \sum_{i\in \mathcal{I}_1} w(X_i)\mathbb{1}{\{V(X_i,1) < V(X_{n+j},0) \}}  }{\sum_{i=1}^n w(X_i) + w(X_{n+j})}  \\
&=\frac{\sum_{i\in \mathcal{I}_0} w(X_i)\mathbb{1}{\{V(X_i,0) < V(X_{n+j},0) \}}   +  w(X_{n+j}) }{\sum_{i=1}^n w(X_i) + w(X_{n+j})} \\ 
&< \frac{\sum_{i\in \mathcal{I}_0} w(X_i)\mathbb{1}{\{V(X_i,0) < V(X_{n+j},0) \}}   +  w(X_{n+j}) }{\sum_{i\in \mathcal{I}_0} w(X_i) + w(X_{n+j})} = p_j',
\end{align*}
where the second lines uses \eqref{eq:monotone_V}. That is, with this nonconformity score (which is shown in \cite{jin2022selection} to be powerful), the p-values constructed in Algorithm \ref{alg:bh} is strictly smaller than those in Algorithm \ref{alg:bh_cond}, leading to larger rejection sets and higher power. Furthermore, in this case,
\begin{align*}
\frac{p_j}{p_j'} = \frac{\sum_{i\in \mathcal{I}_0} w(X_i) + w(X_{n+j})}{\sum_{i=1}^n w(X_i) + w(X_{n+j})}
\end{align*}
is the weighted proportion of negative calibration samples. Thus, the power gain of Algorithm \ref{alg:bh} is more significant when there are more positive samples in the test distribution.

Second, Algorithm \ref{alg:bh_cond} is suitable for dealing with imbalanced data, such as those encountered in drug discovery. For instance, after selecting a subset of molecules or compounds for virtual screening, the experimenter may discard a few samples with $Y=0$ as she would deem them as uninteresting. This bias would make Algorithm \ref{alg:bh} inapplicable. However, if the decision to discard or not does not depend on $X$, the shift between $P_{X\mid Y=0}$ and $Q_{X\mid Y=0}$ remains the same as the covariate shift incurred by selection into screening, and Algorithm \ref{alg:bh_cond} still provides reliable selection.

We finally illustrate the hypothesis-conditional variant from Algorithm \ref{alg:bh_cond} on outlier detection tasks, where there is a covariate shift between calibration inliers and test inliers. We focus on scenarios where the covariate shift is known. For instance, if the demographics in normal financial transactions are adjusted by rejection sampling to balance between male and female, urban and rural users, and so on, then the covariate shift is given by the sampling weights in the adjustment.

### 6.2 Simulation studies

We adapt the simulation setting in \cite{bates2021testing} to a scenario with covariate shift induced by rejection sampling. We fix a test sample size $n_{\textnormal{test}} = 1000$ and calibration sample size $n_\textnormal{calib}=1000$.

At the beginning of the experiment, we sample a subset $\mathcal{W}\subseteq \mathbb{R}^{50}$ with $|\mathcal{W}|=50$, where each element in $\mathcal{W}$ is independently drawn from $\textrm{Unif}([-3,3]^{50})$, and hold it as fixed afterwards. Fix a proportion of outliers at $\rho_{\texttt{pop}}\in \{0.1,0.2,\dots,0.5\}$. The number of outliers in the test data is $n_\textnormal{test} \cdot \rho_{\texttt{pop}}$, where each of them is i.i.d. generated as $X_{n+j}=\sqrt{a}V_{n+j}+W_{n+j}$ for signal strength $a$ varying in the range $[1,4]$, $V_{n+j}\sim N(0,I_{50})$, and $W_{n+j}\sim \textrm{Unif}(\mathcal{W})$. Following this, the test inliers are i.i.d. generated as $X_{n+j}= V_{n+j}+W_{n+j}$, whose distribution is denoted as $Q_X$. The calibration inliers are i.i.d. generated from $P_X$ with $\mathrm{d} Q_X/\mathrm{d} P_X(x)=w(x)\propto \sigma(x^\top\theta)$ ($\sigma(\cdot)$ is the sigmoid), where $\theta\in \mathbb{R}^{50}$ and $\theta_j=0.1\cdot\mathbb{1}\{j\leq 5\}$. We also generate $n_{\textnormal{train}}=1000$ training sample from $P_X$. This setting mimics a stylized scenario where the calibration inliers are collected in a way such that the preference is prescribed by a logistic function of $X$, and is known to the data analyst.

We train a one-class SVM $\widehat{\mu}(x)$ with `rbf` kernel using the `scikit-learn` Python library, and apply Algorithm \ref{alg:bh_cond} with $Z_i=X_i$ at FDR level $q=0.1$ using all the three pruning methods. All procedures are repeated $N=1000$ times, and the FDPs and proportions of true discoveries are averaged to estimate the FDR and power. We also evaluate the BH procedure applied to p-values constructed in Line 4 of Algorithm \ref{alg:bh_cond}. Figure \ref{fig:fdr_outlier} shows the empirical FDR across runs. In line with Theorem \ref{thm:fdr_cond}, we see that the FDR is always below $(1-\rho_{\texttt{pop}})q$. Also, BH applied to weighted conformal p-values empirically controls the FDR and demonstrates comparable performance.

**Figure 11 Caption:** Empirical FDR averaged over $N=1000$ runs under increasing levels $a$ of signal strength. Each subplot corresponds to one value of $\rho_{\texttt{pop}}$. The red dashed lines are at the nominal level $q=0.1$.
\label{fig:fdr_outlier}

Figure \ref{fig:power_outlier} plots the power of the four methods. Interestingly, although they differ in FDR, especially when the signal strength is small, they achieve nearly identical power, finding about the same number of true discoveries.

**Figure 12 Caption:** Empirical power averaged over $N=1000$ runs. Everything else is as in Figure \ref{fig:fdr_outlier}.
\label{fig:power_outlier}

### 6.3 Real data application

We then apply Algorithm \ref{alg:bh_cond} to a bank marketing dataset \cite{moro2014data}; this dataset has been used to benchmark outlier detection algorithms \cite{pang2019deep,pang2021deep}. In short, each sample in the dataset represents a phone contact to a potential client, whose demographic information such as age, gender, and marital status is also recorded in the features. The outcome is a binary indicator $Y$, where $Y=1$ represents a successful compaign. Positive samples are relatively rare, accounting for about $10\%$ of all records.

In our experiments, we always use negative samples as the calibration data. We find this dataset interesting because the classification and outlier detection perspectives are somewhat blurred here. Viewing this as a classification task, Example \ref{ex:binary} becomes relevant, and our theory implies FDR control as long as the distribution of negative samples (i.e., that of $X$ given $Y=0$) admits the covariate shift. In the sense of finding positive responses (so as to reach out to these promising clients), controlling the FDR ensures efficient use of campaign resources. Alternatively, from an outlier detection perspective, those $Y=1$ are outliers \cite{pang2019deep} that deserve more attention. Taking either of the two perspectives leads to the same calibration set and the same guarantees following Section \ref{subsec:hypo_cond_fdr}. Perhaps the only methodological distinction is whether positive samples are leveraged in the training process. Similar considerations also appeared in \cite{liang2022integrative}, but the scenario therein is more sophisticated than ours as they also use known outliers (positive samples) in constructing p-values. As a classification problem, we may use positive samples to train a classifier and produce nonconformity scores. As an outlier detection problem, however, the widely-used one-class classification relies exclusively on inliers (negative samples). We will evaluate the performance of both (i.e., train the nonconformity score with or without positive samples); see procedures `cond_class` and `outlier` below.

For comparison, we additionally evaluate Algorithm \ref{alg:bh} which operates under a covariate shift assumption on the joint distribution of $(X,Y)$ (see procedure `sup_class` below); this is in contrast to Algorithm \ref{alg:bh_cond} which puts no assumption on positive samples.

The total sample size is $N=41188$ with $4640$ positive samples. We use rejection sampling to create the covariate shift between training/calibration and test samples. In details, we use a subset of features $X^*\in \mathbb{R}^4$ representing age and indicators of marriage, basic four-year education, and housing. We first randomly select a subset of the original dataset as the test data. Each sample enters the test data with probability $e(x):=0.125\, \sigma({\theta^\top x^*})$, where $\theta=(1,1/2,1/2,1/2)^\top$. This creates a covariate shift $w(x)\propto e(x)/(1-e(x))$ between null samples in calibration and test folds, so that the test set contains more senior, married, well-educated and housed people.

To test the two perspectives, we consider three procedures with FDR levels $q\in \{0.2, 0.5, 0.8\}$:
1.  `sup_class`: We randomly split the data that is not in the test fold into two equally-sized halves as the training and calibration folds, $\mathcal{D}_{\textnormal{train}}$ and $\mathcal{D}_{\textnormal{calib}}$. We use $\mathcal{D}_{\textnormal{train}}$ to train an SVM classifier using the `scikit-learn` Python library with `rbf` kernel. Then, we apply Algorithm \ref{alg:bh} with $V(x,y)=100\cdot y - \widehat{\mu}(x)$, where $\widehat{\mu}(\cdot)$ is the class probability output from the SVM classifier.
2.  `cond_class`: With the same random split as in (i), the first half is used as $\mathcal{D}_{\textnormal{train}}$, while we set $\mathcal{D}_{\textnormal{calib}}$ as the set of negative samples in the second half. We use $\mathcal{D}_{\textnormal{train}}$ to train an SVM classifier in the same way as in (i) to obtain the predicted class probability $\widehat{\mu}(\cdot)$. Lastly, we apply Algorithm \ref{alg:bh_cond} with $V(x)=-\widehat{\mu}(x)$.
3.  `outlier`: With the same split, we set $\mathcal{D}_{\textnormal{train}}$ and $\mathcal{D}_{\textnormal{calib}}$ as the two folds after discarding all the negative samples. We use $\mathcal{D}_{\textnormal{train}}$ to train a one-class SVM classifier using the `scikit-learn` Python library with `rbf` kernel and obtain the output $\widehat{\mu}(\cdot)$. We then apply Algorithm \ref{alg:bh_cond} with $V(x)=-\widehat{\mu}(x)$.

To ensure consistency, in all procedures, the SVM-based classifier uses the parameter `gamma`$=0.01$.
FDPs and power (proportions of true discoveries) across $N=1000$ independent runs are shown in Figures \ref{fig:real_outlier_fdr} and \ref{fig:real_outlier_power}, respectively.

**Figure 13 Caption:** FDPs over $N=1000$ independent data splits. Each subplot corresponds to one nominal FDR level shown by means of red dashed lines.
\label{fig:real_outlier_fdr}

In Figure \ref{fig:real_outlier_fdr}, we observe FDR control for all the methods in all settings. Across the four testing methods, the FDPs are very similar, hence all of them are reasonable choices. However, both the values and the variability of the FDP vary across settings. The methods `cond_class` and `outlier` from Algorithm \ref{alg:bh_cond} do not use all of the error budget—note the $\pi_0$ factor in Theorem \ref{thm:fdr_cond}. In contrast, Algorithm \ref{alg:bh} leverages super-population structure (which is present in this problem) and leads to tight FDPs and FDR around the target level. In all cases, the outlier detection approach, where only negative samples are used in the training process, has more variable FDPs. With this dataset, the variability of FDP around the FDR is visible for $q=0.2$, but it decreases as we increase the FDR level.

**Figure 14 Caption:** Proportion of true discoveries (empirical power). Everything else is as in Figure \ref{fig:real_outlier_fdr}.
\label{fig:real_outlier_power}

In Figure \ref{fig:real_outlier_power}, the power across the four multiple testing methods is similar. However, we observe drastic differences among the three settings. While using the same classifier, `cond_class` has lower power than `sup_class`, per our discussion in Section \ref{subsubsec:compare}. We also see that `outlier` has much lower power, which may be due to the fact that the nonconformity score is not sufficiently powerful in distinguishing inliers from outliers as the training process does not use outliers. Our results suggest that in outlier detection problems, even when we do not want to impose any distributional assumption on outliers (so that Algorithm \ref{alg:bh} becomes less reasonable), utilizing known outliers may be helpful in obtaining better nonconformity scores.

Finally, we observe negligible difference between the rejection sets returned by the BH procedure and Algorithm \ref{alg:bh}. There at most five distinct decisions among $\approx 3700$ test samples.

## Acknowledgement

The authors thank John Cherian, Issac Gibbs, Kevin Guo, Kexin Huang, Jayoon Jang, Lihua Lei, Shuangning Li, Zhimei Ren, Hui Xu, and Qian Zhao for helpful discussions.
E.C. and Y.J. were supported by the Office of Naval Research grant N00014-20-1-2157, the National Science Foundation grant DMS-2032014, the Simons Foundation under award 814641, and the ARO grant 2003514594.