TOPDIR=`pwd`
HMCDIR=${TOPDIR}/pybin
TESTDIR=${TOPDIR}/testing

if [ ! -d ${TESTDIR} ];
then
    mkdir -p ${TESTDIR}
fi

PYTHON=python
