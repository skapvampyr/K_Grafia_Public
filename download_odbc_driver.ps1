# Descargar e instalar el driver ODBC para SQL Server
Write-Host "Descargando..."
Invoke-WebRequest 'https://go.microsoft.com/fwlink/?linkid=2122167' -OutFile 'msodbcsql.msi'

Write-Host "Instalando msodbcsql..."
Start-Process 'msiexec.exe' -ArgumentList '/i msodbcsql.msi IACCEPTMSODBCSQLLICENSETERMS=YES /quiet' -NoNewWindow -Wait

# Descargar e instalar las herramientas de línea de comandos de SQL Server (sqlcmd y bcp)
Invoke-WebRequest 'https://go.microsoft.com/fwlink/?linkid=2122794' -OutFile 'msodbcsql.msi'

Write-Host "Instalando mssql-tools..."
Start-Process 'msiexec.exe' -ArgumentList '/i mssql-tools.msi IACCEPTMSSQLTOOLSLICENSETERMS=YES /quiet' -NoNewWindow -Wait

# Agregar sqlcmd y bcp al PATH
$env:Path += ";C:\Program Files\Microsoft SQL Server\Client SDK\ODBC\170\Tools\Binn\"

Write-Host "HECHO"