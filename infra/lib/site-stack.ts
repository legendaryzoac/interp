import {
  CfnOutput,
  Duration,
  Stack,
  type StackProps,
} from 'aws-cdk-lib'
import * as acm from 'aws-cdk-lib/aws-certificatemanager'
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront'
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins'
import * as iam from 'aws-cdk-lib/aws-iam'
import * as route53 from 'aws-cdk-lib/aws-route53'
import * as targets from 'aws-cdk-lib/aws-route53-targets'
import * as s3 from 'aws-cdk-lib/aws-s3'
import type { Construct } from 'constructs'

const DOMAIN = 'interp.zackwithers.com'
const ZONE_ID = 'Z0874780F3FVBPDZKEOR'
const ZONE_NAME = 'zackwithers.com'
const GITHUB_REPO = 'legendaryzoac/interp'

export class SiteStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props)

    // Attribute lookup (not fromLookup) keeps synth credential-free — the zone
    // id is stable, so no context cache or account access is needed.
    const zone = route53.HostedZone.fromHostedZoneAttributes(this, 'Zone', {
      hostedZoneId: ZONE_ID,
      zoneName: ZONE_NAME,
    })

    // CloudFront requires the cert in us-east-1; the whole stack lives there.
    const certificate = new acm.Certificate(this, 'Certificate', {
      domainName: DOMAIN,
      validation: acm.CertificateValidation.fromDns(zone),
    })

    const siteBucket = new s3.Bucket(this, 'SiteBucket', {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
    })

    const distribution = new cloudfront.Distribution(this, 'Distribution', {
      defaultRootObject: 'index.html',
      domainNames: [DOMAIN],
      certificate,
      minimumProtocolVersion: cloudfront.SecurityPolicyProtocol.TLS_V1_2_2021,
      httpVersion: cloudfront.HttpVersion.HTTP2_AND_3,
      priceClass: cloudfront.PriceClass.PRICE_CLASS_100,
      defaultBehavior: {
        origin: origins.S3BucketOrigin.withOriginAccessControl(siteBucket),
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        compress: true,
      },
      // SPA routing: unknown paths fall through to index.html
      errorResponses: [
        {
          httpStatus: 403,
          responseHttpStatus: 200,
          responsePagePath: '/index.html',
          ttl: Duration.minutes(1),
        },
        {
          httpStatus: 404,
          responseHttpStatus: 200,
          responsePagePath: '/index.html',
          ttl: Duration.minutes(1),
        },
      ],
    })

    new route53.ARecord(this, 'AliasRecord', {
      zone,
      recordName: 'interp',
      target: route53.RecordTarget.fromAlias(
        new targets.CloudFrontTarget(distribution),
      ),
    })

    new route53.AaaaRecord(this, 'AliasRecordIpv6', {
      zone,
      recordName: 'interp',
      target: route53.RecordTarget.fromAlias(
        new targets.CloudFrontTarget(distribution),
      ),
    })

    // Deploy role assumed by GitHub Actions via the account's existing
    // OIDC provider; scoped to pushes on this repo's main branch.
    const ciRole = new iam.Role(this, 'GithubDeployRole', {
      roleName: 'interp-github-deploy',
      assumedBy: new iam.WebIdentityPrincipal(
        `arn:aws:iam::${this.account}:oidc-provider/token.actions.githubusercontent.com`,
        {
          StringEquals: {
            'token.actions.githubusercontent.com:aud': 'sts.amazonaws.com',
            'token.actions.githubusercontent.com:sub': `repo:${GITHUB_REPO}:ref:refs/heads/main`,
          },
        },
      ),
    })
    siteBucket.grantPut(ciRole)
    siteBucket.grantDelete(ciRole)
    ciRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ['s3:ListBucket'],
        resources: [siteBucket.bucketArn],
      }),
    )
    distribution.grantCreateInvalidation(ciRole)

    new CfnOutput(this, 'BucketName', { value: siteBucket.bucketName })
    new CfnOutput(this, 'DistributionId', { value: distribution.distributionId })
    new CfnOutput(this, 'CiRoleArn', { value: ciRole.roleArn })
    new CfnOutput(this, 'Url', { value: `https://${DOMAIN}` })
  }
}
