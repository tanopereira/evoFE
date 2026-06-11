library(mlrMBO)
library(ParamHelpers)
library(smoof)

ps <- makeParamSet(
  makeNumericParam("x", lower = 0, upper = 10),
  makeNumericParam("y", lower = 0, upper = 10, tunable = FALSE, default = 5)
)

fn <- function(x) {
  print(x)
  (x$x - 2)^2 + (x$y - 5)^2
}

obj <- makeSingleObjectiveFunction(name = "test", fn = fn, par.set = ps, has.simple.signature = FALSE)

des <- generateDesign(n = 5, par.set = ps)
print(des)
