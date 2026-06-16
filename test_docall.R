f <- function(x = 1, y = 2) { x + y }
print(do.call(f, c(list(x = 5), list(x = 10))))
