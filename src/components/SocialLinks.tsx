type SocialLinksProps = {
  className?: string;
  showDomains?: boolean;
  size?: "sm" | "md";
};

type IconProps = {
  className?: string;
};

const GitHubIcon = ({ className }: IconProps) => (
  <svg
    viewBox="0 0 24 24"
    fill="currentColor"
    aria-hidden="true"
    className={className}
  >
    <path d="M12 2C6.477 2 2 6.485 2 12.012c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.866-.013-1.699-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.031 1.531 1.031.892 1.529 2.341 1.088 2.91.832.091-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.953 0-1.094.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844a9.55 9.55 0 0 1 2.504.337c1.909-1.296 2.748-1.026 2.748-1.026.546 1.378.202 2.397.1 2.65.64.7 1.028 1.594 1.028 2.688 0 3.85-2.338 4.697-4.566 4.945.359.309.678.92.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .268.18.58.688.481A10.015 10.015 0 0 0 22 12.012C22 6.485 17.523 2 12 2Z" />
  </svg>
);

const LinkedInIcon = ({ className }: IconProps) => (
  <svg
    viewBox="0 0 24 24"
    fill="currentColor"
    aria-hidden="true"
    className={className}
  >
    <path d="M4.98 3.5A2.5 2.5 0 1 1 5 8.5a2.5 2.5 0 0 1-.02-5ZM3 9h4v12H3zM9 9h3.8v1.64h.05c.53-1 1.83-2.06 3.77-2.06C20.4 8.58 22 10.52 22 14.1V21h-4v-6.1c0-1.46-.03-3.34-2.03-3.34-2.03 0-2.34 1.58-2.34 3.23V21H9z" />
  </svg>
);

const links = [
  {
    key: "github",
    href: "https://saiprapul.github.io",
    label: "GitHub",
    domain: "saiprapul.github.io",
    Icon: GitHubIcon
  },
  {
    key: "linkedin",
    href: "https://linkedin.com/in/saiprapul-r-thotapally",
    label: "LinkedIn",
    domain: "linkedin.com/in/saiprapul-r-thotapally",
    Icon: LinkedInIcon
  }
];

const SocialLinks = ({ className, showDomains = false, size = "md" }: SocialLinksProps) => {
  const iconSize = size === "sm" ? "h-4 w-4" : "h-5 w-5";
  const textSize = size === "sm" ? "text-sm" : "text-sm";

  return (
    <div className={`flex flex-wrap items-center gap-3 ${className ?? ""}`}>
      {links.map(({ key, href, label, domain, Icon }) => (
        <a
          key={key}
          href={href}
          target="_blank"
          rel="noreferrer"
          className={`inline-flex items-center gap-2 ${textSize} text-muted transition hover:text-white`}
          aria-label={label}
        >
          <Icon className={iconSize} />
          <span className="truncate">{showDomains ? domain : label}</span>
        </a>
      ))}
    </div>
  );
};

export default SocialLinks;
