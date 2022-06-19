#!/bin/bash

# definitions
PYBIN=python3
PYENV=env/crf
PYENV_OPTS="--copies"
ENV_PYTHON=${PYENV}/bin/python3
ENV_PIP=${PYENV}/bin/pip3
MODULES="sklearn-crfsuite scikit-learn seqeval"
LOGFILE=deploy.log

# functions
die()
{
    echo $*
    exit 1
}

check()
{
    if [ $? -eq 0 ]; then
        echo "ok"
        echo -e "installation result: ok" >> ${LOGFILE}
    else
        echo "failed"
        echo -e "installation result: failed" >> ${LOGFILE}
    fi
}

plog()
{
    echo -e "$*"
    echo -e "$*" >> ${LOGFILE}
}
#----------------------------------------

# init logfile
echo "DEPLOY on `date`" > ${LOGFILE}
echo -e "-----------------------------------------\n\n" >> ${LOGFILE}

plog "Creating python environment..."
echo -n "   creating pyenv     "
${PYBIN} -m venv ${PYENV_OPTS} ${PYENV} >> ${LOGFILE} 2>&1
check

echo -n "   pip upgrade     "
${ENV_PIP} install --upgrade pip >> ${LOGFILE} 2>&1
check

plog "\nInstalling modules"
for package in ${MODULES}
do
    echo -n "   ${package}...     "
    ${ENV_PIP} install --upgrade ${package} >> ${LOGFILE} 2>&1
    check
done
