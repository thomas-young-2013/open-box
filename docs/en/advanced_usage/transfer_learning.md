# Transfer Learning

When performing BBO, users often run tasks that are similar to
previous ones. This fact can be used to speed up the current task.
Compared with Vizier, which only provides limited transfer learning
functionality for single-objective BBO problems, OpenBox employs
a general transfer learning framework with the following
advantages:

1) support for the generalized black-box optimization
problems

2) compatibility with most BO methods.
OpenBox takes as input observations from 𝐾 + 1 tasks: 𝐷1, ...,
𝐷𝐾 for 𝐾 previous tasks and 𝐷𝑇 for the current task. Each 𝐷𝑖 = {(𝒙𝑖𝑗, 𝒚𝑖𝑗)}𝑛𝑖
𝑗=1, 𝑖 = 1, ...,𝐾, includes a set of observations. Note that,
𝒚 is an array, including multiple objectives for configuration 𝒙.
For multi-objective problems with 𝑝 objectives, we propose to
transfer the knowledge about 𝑝 objectives individually. Thus, the
transfer learning of multiple objectives is turned into 𝑝 single-objective
transfer learning processes. For each dimension of the
objectives, we take the following transfer-learning technique. 1)
We first train a surrogate model 𝑀𝑖 on 𝐷𝑖 for the 𝑖𝑡ℎ prior task
and 𝑀𝑇 on 𝐷𝑇 ; based on 𝑀1:𝐾 and 𝑀𝑇 , we then build a transfer
learning surrogate by combining all base surrogates:
𝑀TL = agg({𝑀1, ...,𝑀𝐾,𝑀𝑇 };w);

3) the surrogate 𝑀TL is used to guide the configuration search,
instead of the original 𝑀𝑇 . Concretely, we use gPoE to combine
the multiple base surrogates (agg), and the parameters w are calculated
based on the ranking of configurations, which reflects the similarity
between the source tasks and the target task.


## Performance Rank

<p align="center">
<img src="https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/imgs/tl_lightgbm_75_rank_result.svg" width="70%">
</p>

+ Average rank of tuning LightGBM with transfer learning
