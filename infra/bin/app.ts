import { App } from 'aws-cdk-lib'
import { SiteStack } from '../lib/site-stack'

const app = new App()

new SiteStack(app, 'InterpSite', {
  env: { account: '545628619410', region: 'us-east-1' },
})
