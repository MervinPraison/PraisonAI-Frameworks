/**
 * PraisonAI-Frameworks gate configuration (hub mode, tools profile base).
 */
module.exports = {
  repoFullName: 'MervinPraison/praisonai-frameworks',
  gitUser: 'MervinPraison',
  gitEmail: 'MervinPraison@users.noreply.github.com',
  triggerLogins: ['MervinPraison', 'github-actions[bot]'],
  productPathPrefixes: ['src/praisonai_frameworks/', 'tests/'],
  sensitivePathPatterns: [/^\.github\/workflows\//],
  requiredCheckPatterns: [/^ci$/i, /python/i, /test/i, /lint/i, /ruff/i],
  optionalCancelledChecks: ['detect-and-trigger'],
  optionalCancelledWhenCoreGreen: [],
  ciWorkflowFile: 'ci.yml',
  ciWorkflowName: 'CI',
  claudeWorkflowName: 'Claude Assistant',
  mergeGateWorkflowRuns: ['CI', 'Claude Assistant'],
  ciFailureWorkflowRuns: ['CI'],
  testCommand: 'pytest tests/unit -q',
  docsUrl: 'https://docs.praison.ai/features/framework-adapters',
  architectureDoc: 'AGENTS.md',
  pypiPackageName: 'praisonai-frameworks',
  finalClaudeScope:
    'SCOPE: Focus ONLY on PraisonAI-Frameworks (third-party framework adapters). Not core SDK, wrapper CLI, or tools/plugins repos.',
  finalClaudeProductValue:
    '4. Adapter value: minimal adapter surface, lazy optional deps, entry-point registration; no core Agent/runtime duplication.',
  mergeGateProductValue:
    'Confirm adapter strengthens praisonai-frameworks without scope creep or core SDK duplication.',
  mergeGateLayering:
    'BLOCK if agent runtime, protocols, or CLI registry logic was added here instead of praisonaiagents/praisonai.',
  agentPyChecks: false,
  reviewBotLogins: ['coderabbitai[bot]', 'qodo-code-review[bot]', 'greptile-apps[bot]'],
  externalRepos: [],
};
