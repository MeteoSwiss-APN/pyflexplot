Move from regular version to dev version::
  bumpversion --allow-dirty --verbose --no-commit --new-version=1.0.6.dev0 dummy

Bump build number of dev version::
  bumpversion --allow-dirty --verbose --no-commit build

Release dev version, returning to regular version number::
  bumpversion --allow-dirty --verbose --commit --tag release
