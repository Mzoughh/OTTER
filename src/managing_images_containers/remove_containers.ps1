Write-Output "Search and remove docker container named as 'module_number_container'"


$containers = docker ps -a --filter "name=^module_[0-9]+_container$" --format "{{.ID}} {{.Names}}"

if ([string]::IsNullOrWhiteSpace($containers)) {
    Write-Output "No corresponding container found"
} else {
    Write-Output "Containers found:"
    Write-Output $containers
    
    Write-Output "Remove these containers"
    $containers -split "`n" | ForEach-Object {
        $container_id = ($_ -split " ")[0]
        docker rm -f $container_id
    }

    Write-Output "Docker containers removed : Ready for a new deployment"
}
