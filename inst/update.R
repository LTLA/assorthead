# Checks for new releases that might be cause for manifest updates. 

get_latest_tag <- function(name, url) {
    tmpname <- file.path("../_sources", name)
    if (!file.exists(tmpname)) {
        stopifnot(system2("git", c("clone", url, tmpname)) == "0")
    } else {
        stopifnot(system2("git", c("-C", tmpname, "fetch", "--all")) == "0")
    }

    tags <- system2("git", c("-C", tmpname, "tag"), stdout=TRUE)

    if (name == "Eigen") {
        tags <- tags[grepl("^[0-9]+\\.[0-9]+\\.[0-9]+$", tags)]
        sanitized <- tags
    } else {
        tags <- tags[grepl("^v[0-9]+\\.[0-9]+\\.[0-9]+$", tags)]
        sanitized <- substr(tags, 2, nchar(tags))
    }

    tags[which.max(package_version(sanitized))]
}

tab <- read.csv("manifest.csv")
for (i in seq_len(nrow(tab))) {
    tab$version[i] <- get_latest_tag(tab$name[i], tab$url[i])
}

write.csv(tab[0,], file="manifest.csv", row.names=FALSE, quote=FALSE)
write.table(tab, file="manifest.csv", col.names=FALSE, row.names=FALSE, quote=4, append=TRUE, sep=",")
