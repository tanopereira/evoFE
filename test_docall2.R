f <- function(x = 1, y = 2) { x + y }
print(do.call(f, list(y = 10)))
