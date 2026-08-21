#!/usr/bin/R 

dir.create("licenses", showWarnings=FALSE)

# Saving these in the parent directory to avoid build/installation warnings.
dir.create("../_versions", showWarnings=FALSE)
dir.create("../_sources", showWarnings=FALSE) 

get_version_file <- function(name) {
    file.path("../_versions", name)
}

already_exists <- function(name, version) {
    vfile <- get_version_file(name)
    if (file.exists(vfile)) {
        existing_version <- readLines(vfile)
        if (existing_version == version) {
            return(TRUE)
        }
    }
    return(FALSE)
}

git_clone <- function(name, url, version) {
    tmpname <- file.path("../_sources", name)
    if (!file.exists(tmpname)) {
        stopifnot(system2("git", c("clone", url, tmpname)) == "0")
    } else {
        stopifnot(system2("git", c("-C", tmpname, "fetch", "--all")) == "0")
#        stopifnot(system2("git", c("-C", tmpname, "pull")) == "0")
    }
    stopifnot(system2("git", c("-C", tmpname, "checkout", version)) == "0")
    return(tmpname)
}

dir_copy <- function(from, to, ...) {
    for (f in list.files(from, ..., recursive=TRUE)) {
        dir.create(file.path(to, dirname(f)), recursive=TRUE, showWarnings=FALSE)
        file.copy(file.path(from, f), file.path(to, f))
    }
}

manifest <- read.csv("manifest.csv")

##################################################

for (i in seq_len(nrow(manifest))) {
    name <- manifest$name[i]
    url <- manifest$url[i]
    version <- manifest$version[i]

    if (name %in% c("annoy", "hnswlib", "Eigen")) {
        next
    }

    if (already_exists(name, version)) {
        cat(name, " (", version, ") is already present\n", sep="")
        next
    }

    tmpname <- git_clone(name, url, version)
    include.path <- file.path("include", name)
    unlink(include.path, recursive=TRUE)
    dir_copy(file.path(tmpname, include.path), include.path)

    license.path <- file.path("licenses", name)
    unlink(license.path, recursive=TRUE)
    dir.create(license.path, recursive=TRUE)
    file.copy(file.path(tmpname, "LICENSE"), license.path)

    vfile <- get_version_file(name)
    write(file=vfile, version)
}

####################################################

(function() {
    name <- "annoy"
    i <- which(manifest$name == name)
    version <- manifest$version[i]
    url <- manifest$url[i]

    if (already_exists(name, version)) {
        cat(name, " (", version, ") is already present\n", sep="")
        return(NULL)
    }

    tmpname <- git_clone(name, url, version)
    include.path <- file.path("include", name)
    unlink(include.path, recursive=TRUE)
    dir_copy(file.path(tmpname, "src"), include.path, pattern="\\.h$")

    license.path <- file.path("licenses", name)
    unlink(license.path, recursive=TRUE)
    dir.create(license.path, recursive=TRUE)
    file.copy(file.path(tmpname, "LICENSE"), license.path)

    vfile <- get_version_file(name)
    write(file=vfile, version)
})()

####################################################

(function() {
    name <- "hnswlib"
    i <- which(manifest$name == name)
    version <- manifest$version[i]
    url <- manifest$url[i]

    if (already_exists(name, version)) {
        cat(name, " (", version, ") is already present\n", sep="")
        return(NULL)
    }

    tmpname <- git_clone(name, url, version)
    include.path <- file.path("include", name)
    unlink(include.path, recursive=TRUE)
    dir_copy(file.path(tmpname, "hnswlib"), include.path)

    license.path <- file.path("licenses", name)
    unlink(license.path, recursive=TRUE)
    dir.create(license.path, recursive=TRUE)
    file.copy(file.path(tmpname, "LICENSE"), license.path)

    vfile <- get_version_file(name)
    write(file=vfile, version)
})()

####################################################

(function() {
    name <- "Eigen"
    i <- which(manifest$name == name)
    version <- manifest$version[i]
    url <- manifest$url[i]

    if (already_exists(name, version)) {
        cat(name, " (", version, ") is already present\n", sep="")
        return(NULL)
    }

    tmpname <- git_clone(name, url, version)
    include.path <- file.path("include", name)
    unlink(include.path, recursive=TRUE)
    dir_copy(file.path(tmpname, "Eigen"), include.path)

    license.path <- file.path("licenses", name)
    unlink(license.path, recursive=TRUE)
    dir.create(license.path, recursive=TRUE)
    dir_copy(file.path(tmpname), license.path, pattern="^COPYING\\.")

    vfile <- get_version_file(name)
    write(file=vfile, version)
})()
