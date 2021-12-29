# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
source /opt/intel/bin/compilervars.sh intel64
source /opt/intel/mkl/bin/mklvars.sh intel64
export PATH=$PATH:/opt/app/vasp.5.4.1/bin/

export PATH=/home/iap13/app/qe-6.4.1/bin/:$PATH
export DISPLAY=":0.0"
export QT_QPA_PLATFORM='offscreen'

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/iap13/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/iap13/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/iap13/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/iap13/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
ulimit -s unlimited
