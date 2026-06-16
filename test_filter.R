library(ParamHelpers)
ps <- makeParamSet(
  makeNumericParam("x", lower = 0, upper = 10),
  makeNumericParam("y", lower = 0, upper = 10, tunable = FALSE)
)
ps_tunable <- filterParams(ps, tunable = TRUE)
print(ps_tunable)
