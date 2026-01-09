# 查看我的仓库和官方仓库的差异
## 拉最新官方仓库
git remote add upstream git@github.com:algorithmicsuperintelligence/openevolve.git
git fetch upstream

## git diff upstream/main HEAD --stat -- . ':!experiments'
 .gitignore                                         |    6 +
 main.py                                            |  361 +
 modification.md                                    |    0
 openevolve/controller.py                           |    7 +-
 openevolve/process_parallel.py                     |    7 +-
 openevolve/prompt/templates.py                     |    4 +-
 scripts/visualizer_step.py                         |  226 +
 src/
 experiments/



## git diff upstream/main HEAD -- openevolve/controller.py openevolve/process_parallel.py openevolve/prompt/templates.py > changes.txt
