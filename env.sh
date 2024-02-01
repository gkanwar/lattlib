TOPDIR="`pwd`"
HMCDIR="${TOPDIR}"/pybin
TESTDIR="${TOPDIR}"/testing
PRODDIR="${TOPDIR}"/prod

if [ ! -d "${TESTDIR}" ];
then
    mkdir -p "${TESTDIR}"
fi

if [ ! -d "${PRODDIR}" ];
then
    mkdir -p "${PRODDIR}"
fi

PYTHON=python
