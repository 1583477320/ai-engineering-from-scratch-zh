# 智能体规则

## startup/state-file-fresh
- category: startup
- check: state_file_fresh
智能体在任何工具调用之前必须读取 agent_state.json。

## forbidden/no-release-script-edits
- category: forbidden
- check: no_release_script_edits
未经批准的发布任务，不得编辑 scripts/release.sh。

## done/tests-pass
- category: definition_of_done
- check: tests_pass
任务完成的唯一标准是验收命令退出码为 0。

## uncertainty/open-question-note
- category: uncertainty
- check: opened_question_when_unsure
置信度低于阈值时，写一个问答笔记，而不是猜测。

## approval/new-dependency
- category: approval
- check: new_dependency_approved
添加运行时依赖需要人工明确审批。
