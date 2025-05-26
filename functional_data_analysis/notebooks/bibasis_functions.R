library(fda)
library(fda.usc)

smooth.bibasis <- function(sarg, targ, y, fdPars, fdPart, fdnames = NULL, returnMatrix = FALSE) {
  # sarg <- argcheck(sarg)
  # targ <- argcheck(targ)
  ns <- length(sarg)
  nt <- length(targ)
  if (!inherits(y, "matrix") && !inherits(y, "array")) {
    stop("'y' is not of class matrix or class array.")
  }
  ydim <- dim(y)
  if (ydim[1] != ns) {
    stop("Number of rows of Y is not the same length as SARG.")
  }
  if (ydim[2] != nt) {
    stop("Number of columns of Y is not the same length as TARG.")
  }
  if (length(ydim) == 2) {
    nsurf <- 1
    ymat <- matrix(y, ns * nt, 1)
  } else {
    nsurf <- ydim[3]
    ymat <- matrix(0, ns * nt, nsurf)
    for (isurf in 1:nsurf) {
      ymat[, isurf] <- matrix(
        y[, , isurf],
        ns * nt, 1
      )
    }
  }
  fdPars <- fdParcheck(fdPars, nsurf)
  fdobjs <- fdPars$fd
  sbasis <- fdobjs$basis
  snbasis <- sbasis$nbasis - length(sbasis$dropind)
  lambdas <- fdPars$lambda
  Lfds <- fdPars$Lfd
  fdPart <- fdParcheck(fdPart, nsurf)
  fdobjt <- fdPart$fd
  tbasis <- fdobjt$basis
  tnbasis <- tbasis$nbasis - length(tbasis$dropind)
  lambdat <- fdPart$lambda
  Lfdt <- fdPart$Lfd
  if (lambdas < 0) {
    warning("Value of lambdas was negative, 0 used instead.")
    lambdas <- 0
  }
  if (lambdat < 0) {
    warning("Value of lambdat was negative, 0 used instead.")
    lambdat <- 0
  }
  if (is.null(fdnames)) {
    fdnames <- vector("list", 3)
    fdnames[[1]] <- "argument s"
    fdnames[[2]] <- "argument t"
    fdnames[[3]] <- "function"
  }
  sbasismat <- eval.basis(sarg, sbasis, 0, returnMatrix)
  tbasismat <- eval.basis(targ, tbasis, 0, returnMatrix)
  basismat <- kronecker(tbasismat, sbasismat)
  if (ns * nt > snbasis * tnbasis || lambdas > 0 || lambdat >
    0) {
    Bmat <- crossprod(basismat, basismat)
    Dmat <- crossprod(basismat, ymat)
    if (lambdas > 0) {
      penmats <- eval.penalty(sbasis, Lfds)
      Bnorm <- sqrt(sum(c(Bmat)^2))
      pennorm <- sqrt(sum(c(penmats)^2))
      condno <- pennorm / Bnorm
      if (lambdas * condno > 1e+12) {
        lambdas <- 1e+12 / condno
        warning(paste(
          "lambdas reduced to", lambdas,
          "to prevent overflow"
        ))
      }
      Imat <- diag(rep(tnbasis, 1))
      Bmat <- Bmat + lambdas * kronecker(Imat, penmats)
    }
    if (lambdat > 0) {
      penmatt <- eval.penalty(tbasis, Lfdt)
      Bnorm <- sqrt(sum(c(Bmat)^2))
      pennorm <- sqrt(sum(c(penmatt)^2))
      condno <- pennorm / Bnorm
      if (lambdat * condno > 1e+12) {
        lambdat <- 1e+12 / condno
        warning(paste(
          "lambdat reduced to", lambdat,
          "to prevent overflow"
        ))
      }
      Imat <- diag(rep(snbasis, 1))
      Bmat <- Bmat + lambdat * kronecker(penmatt, Imat)
    }
    Bmat <- (Bmat + t(Bmat)) / 2
    Lmat <- chol(Bmat)
    Lmatinv <- solve(Lmat)
    Bmatinv <- Lmatinv %*% t(Lmatinv)
    y2cMap <- Bmatinv %*% t(basismat)
    BiB0 <- Bmatinv %*% Bmat
    df <- sum(diag(BiB0))
    coef <- solve(Bmat, Dmat)
    if (nsurf == 1) {
      coefmat <- matrix(coef, snbasis, tnbasis)
    } else {
      coefmat <- array(0, c(snbasis, tnbasis, nsurf))
      for (isurf in 1:nsurf) {
        coefmat[, , isurf] <- matrix(coef[
          ,
          isurf
        ], snbasis, tnbasis)
      }
    }
  } else {
    stop(paste(
      "The number of basis functions exceeds the number of ",
      "points to be smoothed."
    ))
  }
  yhat <- basismat %*% coef
  SSE <- sum((ymat - yhat)^2)
  N <- ns * nt * nsurf
  if (df < N) {
    gcv <- (SSE / N) / ((N - df) / N)^2
  } else {
    gcv <- NA
  }
  bifdobj <- bifd(coefmat, sbasis, tbasis, fdnames)
  smoothlist <- list(
    bifdobj = bifdobj, df = df, gcv = gcv,
    SSE = SSE, penmats = penmats, penmatt = penmatt, y2cMap = y2cMap,
    sarg = sarg, targ = targ, y = y, coef = coefmat
  )
  return(smoothlist)
}
