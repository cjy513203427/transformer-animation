import fs from 'node:fs';
const path='src/features/chapters/NotebookTools.tsx';
const source=fs.readFileSync(path,'utf8');
const header=source.slice(0,source.indexOf('export function RLHFProcess'))
  .replace("import { RLHF_STAGES } from '../../content/notebook';",'')
  .replace('gradientTrajectory, tallyAnswers, trainingBudget','gradientTrajectory, trainingBudget');
fs.writeFileSync('src/features/p1/TrainingTools.tsx',header+source.slice(source.indexOf('export function LearningRateTool')));
fs.unlinkSync(path);
const css=fs.readFileSync('src/styles/notebook.css','utf8');
fs.writeFileSync('src/styles/training-tools.css',css.slice(css.indexOf('.note-tool {')));
fs.unlinkSync('src/styles/notebook.css');
