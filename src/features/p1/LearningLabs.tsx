import P1Lab from './P1Lab';
import { trainingTopic, policyTopic, chainTopic, voteTopic, searchTopic } from '../../content/p1/learning';
export const TrainingLab = () => <P1Lab topic={trainingTopic} />;
export const PolicyLab = () => <P1Lab topic={policyTopic} />;
export const PromptChainLab = () => <P1Lab topic={chainTopic} />;
export const PromptVoteLab = () => <P1Lab topic={voteTopic} />;
export const PromptSearchLab = () => <P1Lab topic={searchTopic} />;
